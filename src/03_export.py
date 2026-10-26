"""Export a trained checkpoint: standard .ply + novel-view orbit video + comparison grid.

The .ply uses the reference 3DGS field layout, so it opens in the usual web
viewers (SuperSplat, antimatter15, etc.) even though it was trained by this
implementation. Our parameters are already stored in exactly the raw form
those viewers expect:
    scale    -> log scale        (viewer applies exp)
    opacity  -> logit            (viewer applies sigmoid)
    rotation -> raw quaternion   (viewer normalises)

Usage:
  python src/03_export.py --ckpt logs/run400/ckpt_last.pt --video --ply --grid
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import model as M
import rasterizer as R

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ckpt(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["state"]
    gm = M.GaussianModel(st["mean3d"].shape[0], st["extent"], dev)
    with torch.no_grad():
        for k in ("mean3d", "log_scale", "quat", "opacity_raw", "sh"):
            getattr(gm, k).copy_(st[k].to(dev))
    gm.sh_degree_active = st["sh_degree_active"]
    return gm, ck


# --------------------------------------------------------------------------
def export_ply(gm, path):
    """Write the reference 3DGS .ply layout (binary little-endian)."""
    n = gm.n
    xyz = gm.mean3d.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    sh = gm.sh.detach().cpu().numpy()                    # [N,16,3]
    f_dc = sh[:, 0, :]                                   # [N,3]
    # channel-major, matching the reference writer
    f_rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)   # [N,45]
    opacity = gm.opacity_raw.detach().cpu().numpy().reshape(n, 1)
    scale = gm.log_scale.detach().cpu().numpy()
    rot = gm.quat.detach().cpu().numpy()

    names = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
             + ["opacity"]
             + [f"scale_{i}" for i in range(3)]
             + [f"rot_{i}" for i in range(4)])

    arr = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scale, rot],
                         axis=1).astype(np.float32)
    assert arr.shape[1] == len(names), (arr.shape, len(names))

    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {m}\n" for m in names)
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())
    return os.path.getsize(path)


# --------------------------------------------------------------------------
def infer_scene_frame(cams):
    """Work out the up-axis, orbit radius and SUPERVISED elevation band.

    Do not hardcode this. NeRF-synthetic uses Blender's +z-up world, and the
    training cameras cover only the UPPER hemisphere - rendering an orbit that
    dips below that band shows the underside of the scene, which no training
    view ever constrained, so it is reconstruction garbage rather than a bug
    in the renderer. Deriving the frame from the data keeps the orbit inside
    the region the model actually learned.
    """
    pos = torch.stack([c.cam_pos for c in cams]).cpu()
    center = torch.zeros(3)                      # these scenes orbit the origin
    rel = pos - center
    r = float(rel.norm(dim=1).mean())

    # the up axis is the one every camera sits on the same side of
    up_axis = 2
    best = -1.0
    for i in range(3):
        lo, hi = float(rel[:, i].min()), float(rel[:, i].max())
        if lo > 0 and abs(float(rel[:, i].mean())) > best:
            best, up_axis = abs(float(rel[:, i].mean())), i

    el = torch.asin(rel[:, up_axis] / rel.norm(dim=1))
    return {"center": center, "radius": r, "up_axis": up_axis,
            "elev_lo": float(el.min()), "elev_hi": float(el.max())}


def orbit_cameras(ref_cams, n_frames, elev_deg=None):
    """A circular orbit around the scene, inside the supervised elevation band."""
    c0 = ref_cams[0]
    dev = c0.cam_pos.device
    fr = infer_scene_frame(ref_cams)
    r, up_axis = fr["radius"], fr["up_axis"]

    if elev_deg is None:
        # sit in the middle of the band the training views actually covered
        el = 0.5 * (fr["elev_lo"] + fr["elev_hi"]) * 0.8
    else:
        el = math.radians(elev_deg)
        el = min(max(el, fr["elev_lo"]), fr["elev_hi"])

    # the two axes orthogonal to "up" span the orbit plane
    a0, a1 = [i for i in range(3) if i != up_axis]
    world_up = torch.zeros(3, device=dev)
    world_up[up_axis] = 1.0

    cams = []
    for i in range(n_frames):
        th = 2 * math.pi * i / n_frames
        eye = torch.zeros(3, device=dev)
        eye[a0] = r * math.cos(el) * math.cos(th)
        eye[a1] = r * math.cos(el) * math.sin(th)
        eye[up_axis] = r * math.sin(el)
        eye = eye + fr["center"].to(dev)

        # our camera basis is (right, down, forward) with right x down = forward
        fwd = torch.nn.functional.normalize(fr["center"].to(dev) - eye, dim=0)
        ref_up = world_up
        if abs(float(torch.dot(fwd, ref_up))) > 0.999:    # looking straight down
            ref_up = torch.zeros(3, device=dev)
            ref_up[a0] = 1.0
        right = torch.nn.functional.normalize(
            torch.cross(fwd, ref_up, dim=0), dim=0)
        down = torch.cross(fwd, right, dim=0)
        Rm = torch.stack([right, down, fwd], dim=0)       # world -> camera
        vm = torch.eye(4, device=dev)
        vm[:3, :3] = Rm
        vm[:3, 3] = -Rm @ eye

        c = D.Camera()
        c.viewmat, c.cam_pos = vm, eye
        c.fx, c.fy, c.cx, c.cy = c0.fx, c0.fy, c0.cx, c0.cy
        c.H, c.W, c.image, c.name = c0.H, c0.W, None, f"orbit{i:04d}"
        cams.append(c)
    return cams, fr, el


@torch.no_grad()
def render_video(gm, cams, out_mp4, fps=30, chunk=16_000_000):
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix="gsplat_orbit_")
    for i, c in enumerate(cams):
        img, _, _ = R.render(gm.mean3d, gm.log_scale, gm.quat, gm.opacity_raw,
                             gm.sh, c.viewmat, c.cam_pos, c.H, c.W,
                             c.fx, c.fy, c.cx, c.cy,
                             sh_degree=gm.sh_degree_active, bg=1.0,
                             chunk_elems=chunk, use_checkpoint=False)
        a = (img.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(a).save(os.path.join(tmp, f"f{i:04d}.png"))
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(cams)} frames", flush=True)
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i",
           os.path.join(tmp, "f%04d.png"), "-c:v", "libx264", "-pix_fmt",
           "yuv420p", "-crf", "18", out_mp4]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_mp4


@torch.no_grad()
def comparison_grid(gm, cams, out_png, n=4, chunk=16_000_000):
    """Render | ground truth | 5x absolute error, stacked for n views."""
    from PIL import Image
    rows = []
    for c in cams[:n]:
        img, _, _ = R.render(gm.mean3d, gm.log_scale, gm.quat, gm.opacity_raw,
                             gm.sh, c.viewmat, c.cam_pos, c.H, c.W,
                             c.fx, c.fy, c.cx, c.cy,
                             sh_degree=gm.sh_degree_active, bg=1.0,
                             chunk_elems=chunk, use_checkpoint=False)
        img = img.clamp(0, 1)
        err = ((img - c.image).abs() * 5).clamp(0, 1)
        rows.append(torch.cat([img, c.image, err], dim=1).cpu().numpy())
    g = (np.concatenate(rows, axis=0) * 255).astype("uint8")
    Image.fromarray(g).save(out_png)
    return out_png


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(HERE, "logs/run400/ckpt_last.pt"))
    p.add_argument("--data", default=os.path.join(HERE, "data/lego"))
    p.add_argument("--out", default=os.path.join(HERE, "logs/export"))
    p.add_argument("--res", type=int, default=400)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--ply", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--grid", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gm, ck = load_ckpt(args.ckpt, dev)
    print(f"loaded iter {ck['iter']} | {gm.n:,} gaussians | "
          f"sh degree {gm.sh_degree_active}")

    if args.ply:
        path = os.path.join(args.out, "scene.ply")
        size = export_ply(gm, path)
        print(f"  ply   -> {path} ({size/1024**2:.1f} MB)")

    if args.grid or args.video:
        test = D.load_blender(args.data, "test", args.res, dev, limit=8)

    if args.grid:
        path = comparison_grid(gm, test, os.path.join(args.out, "compare.png"))
        psnrs = []
        with torch.no_grad():
            for c in test:
                img, _, _ = R.render(gm.mean3d, gm.log_scale, gm.quat,
                                     gm.opacity_raw, gm.sh, c.viewmat,
                                     c.cam_pos, c.H, c.W, c.fx, c.fy, c.cx,
                                     c.cy, sh_degree=gm.sh_degree_active,
                                     bg=1.0, use_checkpoint=False)
                psnrs.append(D.psnr(img.clamp(0, 1), c.image))
        print(f"  grid  -> {path}")
        print(f"  test PSNR {sum(psnrs)/len(psnrs):.2f} dB over {len(psnrs)} views")

    if args.video:
        cams, fr, el = orbit_cameras(test, args.frames)
        print(f"  scene frame: up=+{'xyz'[fr['up_axis']]} radius={fr['radius']:.2f} "
              f"| training elevation {math.degrees(fr['elev_lo']):.0f}.."
              f"{math.degrees(fr['elev_hi']):.0f} deg "
              f"-> orbit at {math.degrees(el):.0f} deg")
        print(f"  rendering {args.frames} orbit frames...")
        path = render_video(gm, cams, os.path.join(args.out, "orbit.mp4"))
        print(f"  video -> {path}")


if __name__ == "__main__":
    main()
