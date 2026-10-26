"""Pack the trained gaussians into a self-contained interactive WebGL viewer.

The point: stop describing the model and let someone orbit it.

The browser viewer runs the SAME math as src/rasterizer.py - project the mean,
push the 3D covariance through the projection Jacobian, draw a 3-sigma quad,
evaluate the gaussian in the fragment shader, composite back-to-front - except
it does it at 60 fps instead of 1.6 s/frame, because it rasterises hardware
triangles rather than materialising [tiles, pixels, gaussians] tensors.

Packing: 28 bytes per splat.
    position   3 x float32   (12 B)
    cov3d      6 x float16   (12 B)   scaled so fp16 has headroom
    colour+a   4 x uint8     ( 4 B)
Splats are ranked by opacity x projected area, so a budget keeps the ones that
actually carry the image.
"""

import argparse
import base64
import io
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import model as M
import rasterizer as R

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C0 = 0.28209479177387814


def quat_to_rot_np(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)


def pack(ck, budget):
    st = ck["state"]
    xyz = st["mean3d"].numpy().astype(np.float32)
    scale = np.exp(st["log_scale"].numpy()).astype(np.float32)
    quat = st["quat"].numpy().astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-st["opacity_raw"].numpy().reshape(-1)))
    sh_dc = st["sh"][:, 0, :].numpy()
    rgb = np.clip(C0 * sh_dc + 0.5, 0.0, 1.0)

    # rank by how much of the image each splat is actually responsible for:
    # opacity x the area of its two largest axes
    s_sorted = np.sort(scale, axis=1)[:, ::-1]
    importance = opacity * s_sorted[:, 0] * s_sorted[:, 1]
    keep = opacity > 0.02
    idx = np.nonzero(keep)[0]
    idx = idx[np.argsort(-importance[idx])][:budget]
    n = idx.size

    xyz, scale, quat = xyz[idx], scale[idx], quat[idx]
    opacity, rgb = opacity[idx], rgb[idx]

    # Sigma = R S S^T R^T, upper triangle (xx, xy, xz, yy, yz, zz)
    Rm = quat_to_rot_np(quat)
    RS = Rm * scale[:, None, :]
    Sigma = RS @ RS.transpose(0, 2, 1)
    cov = np.stack([Sigma[:, 0, 0], Sigma[:, 0, 1], Sigma[:, 0, 2],
                    Sigma[:, 1, 1], Sigma[:, 1, 2], Sigma[:, 2, 2]], axis=1)

    # fp16 goes subnormal below ~6e-5 and these covariances run far smaller,
    # so rescale into a range fp16 represents precisely and undo it in GLSL
    cmax = float(np.abs(cov).max())
    cov_scale = 1024.0 / max(cmax, 1e-20)
    cov16 = (cov * cov_scale).astype(np.float16)

    buf = np.zeros((n, 28), dtype=np.uint8)
    buf[:, 0:12] = xyz.astype(np.float32).view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = cov16.view(np.uint8).reshape(n, 12)
    buf[:, 24:27] = (rgb * 255.0 + 0.5).astype(np.uint8)
    buf[:, 27] = (opacity * 255.0 + 0.5).astype(np.uint8)

    centre = xyz.mean(axis=0)
    return buf.tobytes(), n, cov_scale, centre


def comparison_jpeg(ckpt_path, data_dir, res=400):
    """render | ground truth, as a compact JPEG."""
    from PIL import Image
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["state"]
    gm = M.GaussianModel(st["mean3d"].shape[0], st["extent"], dev)
    with torch.no_grad():
        for k in ("mean3d", "log_scale", "quat", "opacity_raw", "sh"):
            getattr(gm, k).copy_(st[k].to(dev))
    gm.sh_degree_active = st["sh_degree_active"]
    cam = D.load_blender(data_dir, "test", res, dev, limit=1)[0]
    with torch.no_grad():
        img, _, _ = R.render(gm.mean3d, gm.log_scale, gm.quat, gm.opacity_raw,
                             gm.sh, cam.viewmat, cam.cam_pos, cam.H, cam.W,
                             cam.fx, cam.fy, cam.cx, cam.cy,
                             sh_degree=gm.sh_degree_active, bg=1.0,
                             use_checkpoint=False)
    a = (img.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    b = (cam.image.cpu().numpy() * 255).astype("uint8")
    both = np.concatenate([a, b], axis=1)
    bio = io.BytesIO()
    Image.fromarray(both).save(bio, format="JPEG", quality=86, optimize=True)
    return base64.b64encode(bio.getvalue()).decode(), D.psnr(img.clamp(0, 1), cam.image)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(HERE, "logs/run400/ckpt_last.pt"))
    p.add_argument("--data", default=os.path.join(HERE, "data/lego"))
    p.add_argument("--out", default=os.path.join(HERE, "logs/demo.html"))
    p.add_argument("--budget", type=int, default=46000)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw, n, cov_scale, centre = pack(ck, args.budget)
    b64 = base64.b64encode(raw).decode()
    total = ck["state"]["mean3d"].shape[0]
    print(f"packed {n:,} / {total:,} splats -> {len(raw)/1024**2:.2f} MB "
          f"({len(b64)/1024**2:.2f} MB base64)")

    cmp_b64, cmp_psnr = comparison_jpeg(args.ckpt, args.data)
    print(f"comparison jpeg {len(cmp_b64)/1024:.0f} KB, view PSNR {cmp_psnr:.2f} dB")

    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "demo_template.html")).read()
    html = (tpl.replace("__SPLAT_B64__", b64)
               .replace("__SPLAT_N__", str(n))
               .replace("__COV_SCALE__", repr(float(cov_scale)))
               .replace("__CENTRE__", json.dumps([float(x) for x in centre]))
               .replace("__TOTAL__", str(total))
               .replace("__ITER__", str(ck["iter"]))
               .replace("__CMP_B64__", cmp_b64)
               .replace("__CMP_PSNR__", f"{cmp_psnr:.2f}"))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
