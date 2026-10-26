"""Build a point cloud for 3DGS initialisation from a PRE-TRAINED monocular
depth model, to test whether it beats random initialisation.

The hypothesis under test: run a depth network on every training image,
unproject each pixel to 3D, colour it from the pixel, and use that as the
starting cloud instead of random points.

Two things make this non-trivial, and both are handled here.

  1. MONOCULAR DEPTH IS AFFINE-AMBIGUOUS. DepthAnything predicts something
     proportional to INVERSE depth, with an unknown scale and shift PER IMAGE:
         1/Z_i  ~=  a_i * pred_i + b_i
     Unproject each view with its own unknown (a, b) and you get N mutually
     misaligned shells, not one cloud. We solve for every (a_i, b_i) by
     requiring that views AGREE: a 3D point from view i, reprojected into
     view j, must land at the depth view j predicts there. Only the known
     camera poses are used - no ground-truth depth anywhere.

  2. SPLAT COUNT MUST NOT SCALE WITH IMAGE COUNT. One gaussian per pixel per
     image gives 100 x 400 x 400 = 16M points for a scene that needs ~117k.
     Every surface is repeated once per observing camera. We fuse the views
     and voxel-downsample to one point per voxel, which decouples cloud size
     from how many photographs were taken.

Outputs  logs/depth_init/cloud.pt  with xyz [N,3] and rgb [N,3].
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
def predict_depth(cams, paths, dev, model_id="depth-anything/Depth-Anything-V2-Small-hf"):
    """Run the pre-trained depth network on every training view.

    Returns pred [V,H,W] (inverse-depth-like, arbitrary affine) and the
    object mask [V,H,W] taken from the PNG alpha channel.
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    print(f"loading {model_id} ...")
    proc = AutoImageProcessor.from_pretrained(model_id)
    net = AutoModelForDepthEstimation.from_pretrained(model_id).to(dev).eval()

    preds, masks = [], []
    for i, (c, p) in enumerate(zip(cams, paths)):
        raw = Image.open(p)
        alpha = np.asarray(raw.convert("RGBA"), np.float32)[..., 3] / 255.0
        rgb = Image.fromarray(
            (np.asarray(raw.convert("RGBA"), np.float32)[..., :3]).astype(np.uint8))

        with torch.no_grad():
            inp = proc(images=rgb, return_tensors="pt").to(dev)
            out = net(**inp).predicted_depth            # [1,h,w]
            out = F.interpolate(out[:, None], size=(c.H, c.W),
                                mode="bicubic", align_corners=False)[0, 0]
        preds.append(out.float())

        m = torch.from_numpy(alpha).to(dev)[None, None]
        m = F.interpolate(m, size=(c.H, c.W), mode="bilinear", align_corners=False)[0, 0]
        masks.append(m > 0.6)                            # solid object pixels only
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(cams)} views")
    return torch.stack(preds), torch.stack(masks)


# --------------------------------------------------------------------------
def normalise_pred(pred, mask):
    """Map each view's raw prediction to [0,1] over its object pixels.

    Uses 2nd/98th percentiles rather than min/max so a few outlier pixels
    cannot set the range. 1 = nearest surface, 0 = furthest.
    """
    V = pred.shape[0]
    out = torch.zeros_like(pred)
    for v in range(V):
        m = mask[v]
        if m.sum() < 10:
            continue
        vals = pred[v][m]
        lo, hi = vals.quantile(0.02), vals.quantile(0.98)
        out[v] = ((pred[v] - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)
    return out


def depth_from_radius(pnorm, cam_dist, R):
    """Turn normalised relative depth into metric depth given an object radius.

    The key realisation that makes this stable: we ALREADY KNOW how far each
    camera is from the scene centre, because the dataset gives us the poses.
    So the only genuinely unknown quantity is the object's radius R - ONE
    number for the whole scene, not two per image. Depth then interpolates
    between the near and far surface of a sphere of radius R:

        pnorm = 1 (nearest)  ->  Z = cam_dist - R
        pnorm = 0 (furthest) ->  Z = cam_dist + R

    With a single bounded parameter the solution cannot run off to infinity,
    which is exactly what the earlier free-per-image affine fit did.
    """
    return cam_dist - R * (2.0 * pnorm - 1.0)


def multiview_error(cams, pnorm, mask, R, dev, n_pairs=60, n_samp=700, seed=0):
    """Photometric disagreement between views for a given object radius R.

    Lower is better. Uses only the known poses - no ground-truth depth.

    We compare COLOUR, not depth. Comparing depth is degenerate: set R = 0 and
    every point sits at the camera distance, so the views trivially agree
    because there is no depth variation left to disagree about - the sweep
    then happily reports R -> 0 as "best". Colour has no such escape: if the
    depth is wrong the point lands on the wrong part of the object in the
    other view, and the pixel colours stop matching. This is the classic
    multi-view-stereo photo-consistency objective.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    V = len(cams)
    H, W = cams[0].H, cams[0].W
    fx, fy, cx, cy = cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32),
                            indexing="ij")
    dists = torch.stack([c.cam_pos for c in cams]).norm(dim=1)

    errs = []
    for _ in range(n_pairs):
        i, j = torch.randint(0, V, (2,), generator=g).tolist()
        if i == j:
            continue
        sel = torch.nonzero(mask[i].reshape(-1), as_tuple=True)[0]
        if sel.numel() < 50:
            continue
        sel = sel[torch.randint(0, sel.numel(), (n_samp,), device=dev)]

        zi = depth_from_radius(pnorm[i].reshape(-1)[sel], dists[i], R)
        u, v_ = xs.reshape(-1)[sel] + 0.5, ys.reshape(-1)[sel] + 0.5
        pc = torch.stack([(u - cx) * zi / fx, (v_ - cy) * zi / fy, zi], -1)
        Ri, ti = cams[i].viewmat[:3, :3], cams[i].viewmat[:3, 3]
        world = (pc - ti) @ Ri

        Rj, tj = cams[j].viewmat[:3, :3], cams[j].viewmat[:3, 3]
        pj = world @ Rj.T + tj
        zj = pj[:, 2]
        uj = fx * pj[:, 0] / zj.clamp(min=1e-3) + cx
        vj = fy * pj[:, 1] / zj.clamp(min=1e-3) + cy

        inb = (uj > 0) & (uj < W - 1) & (vj > 0) & (vj < H - 1) & (zj > 1e-2)
        if inb.sum() < 30:
            continue
        fl = (vj[inb].long() * W + uj[inb].long()).clamp(0, H * W - 1)
        keep = mask[j].reshape(-1)[fl]        # landed on the object in view j
        if keep.sum() < 30:
            continue
        # photo-consistency: the colour view i saw must match what view j sees
        # at the reprojected pixel
        col_i = cams[i].image.reshape(-1, 3)[sel][inb][keep]
        col_j = cams[j].image.reshape(-1, 3)[fl][keep]
        e = (col_i - col_j).abs().mean(-1)
        errs.append(e.median())           # median = robust to occlusion outliers
    return float(torch.stack(errs).mean()) if errs else float("inf")


def solve_radius(cams, pnorm, mask, dev, lo=0.3, hi=2.5, steps=23):
    """Pick the object radius R that makes the views agree best.

    A 1-D sweep rather than gradient descent: with a single bounded parameter
    this is both cheaper and immune to the divergence that killed the
    free-affine version.
    """
    best, best_e = None, float("inf")
    for R in np.linspace(lo, hi, steps):
        e = multiview_error(cams, pnorm, mask, float(R), dev)
        if e < best_e:
            best, best_e = float(R), e
        print(f"    R = {R:4.2f}   multi-view log-depth error {e:.4f}")
    return best, best_e


# --------------------------------------------------------------------------
def build_cloud(cams, pnorm, mask, R, dev, voxel=0.02, cap=200_000):
    """Unproject every view with the solved radius, fuse, voxel-downsample."""
    H, W = cams[0].H, cams[0].W
    fx, fy, cx, cy = cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32),
                            indexing="ij")
    all_xyz, all_rgb = [], []
    for v, c in enumerate(cams):
        sel = torch.nonzero(mask[v].reshape(-1), as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        dist = c.cam_pos.norm()
        z = depth_from_radius(pnorm[v].reshape(-1)[sel], dist, R)
        u, vv = xs.reshape(-1)[sel] + 0.5, ys.reshape(-1)[sel] + 0.5
        pc = torch.stack([(u - cx) * z / fx, (vv - cy) * z / fy, z], -1)
        # NB: name the rotation Rw, not R - R is the object radius argument,
        # and shadowing it silently corrupted every view after the first.
        Rw, t = c.viewmat[:3, :3], c.viewmat[:3, 3]
        all_xyz.append((pc - t) @ Rw)
        all_rgb.append(c.image.reshape(-1, 3)[sel])
    xyz = torch.cat(all_xyz)
    rgb = torch.cat(all_rgb)
    raw_n = xyz.shape[0]

    return (*voxel_fuse(xyz, rgb, dev, voxel, cap), raw_n)


def voxel_fuse(xyz, rgb, dev, voxel, cap):
    """One point per occupied voxel: this is what removes cross-view duplicates."""
    raw_n = xyz.shape[0]
    key = torch.round(xyz / voxel).long()
    key = key - key.min(0).values
    mul = torch.tensor([1, key[:, 0].max() + 1,
                        (key[:, 0].max() + 1) * (key[:, 1].max() + 1)], device=dev)
    flat = (key * mul).sum(1)
    uniq, inv = torch.unique(flat, return_inverse=True)
    # average the points landing in each voxel
    cnt = torch.zeros(uniq.numel(), device=dev).index_add_(0, inv, torch.ones_like(flat, dtype=torch.float32))
    sx = torch.zeros(uniq.numel(), 3, device=dev).index_add_(0, inv, xyz)
    sc = torch.zeros(uniq.numel(), 3, device=dev).index_add_(0, inv, rgb)
    xyz_v, rgb_v = sx / cnt[:, None], sc / cnt[:, None]

    if xyz_v.shape[0] > cap:
        keep = torch.randperm(xyz_v.shape[0], device=dev)[:cap]
        xyz_v, rgb_v = xyz_v[keep], rgb_v[keep]

    print(f"  raw {raw_n:,} points -> {xyz_v.shape[0]:,} after voxel fusion "
          f"({raw_n / max(xyz_v.shape[0],1):.0f}x reduction)")
    return xyz_v, rgb_v


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "data/lego"))
    p.add_argument("--res", type=int, default=400)
    p.add_argument("--voxel", type=float, default=0.02)
    p.add_argument("--out", default=os.path.join(HERE, "logs/depth_init"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")

    cams = D.load_blender(args.data, "train", args.res, dev)
    import json
    meta = json.load(open(os.path.join(args.data, "transforms_train.json")))
    paths = [os.path.join(args.data, f["file_path"].lstrip("./") + ".png")
             for f in meta["frames"]]
    print(f"{len(cams)} training views")

    pred, mask = predict_depth(cams, paths, dev)
    print(f"depth maps {tuple(pred.shape)}, object pixels/view "
          f"{float(mask.float().sum(dim=(1,2)).mean()):.0f}")

    pnorm = normalise_pred(pred, mask)
    print("solving object radius by multi-view agreement ...")
    R, err = solve_radius(cams, pnorm, mask, dev)
    print(f"  best R = {R:.2f}  (multi-view log-depth error {err:.4f})")

    TARGET = 50_000          # same starting count for every A/B arm

    xyz, rgb, raw_n = build_cloud(cams, pnorm, mask, R, dev, voxel=args.voxel,
                                  cap=TARGET)
    torch.save({"xyz": xyz.cpu(), "rgb": rgb.cpu(), "raw_n": raw_n, "R": R},
               os.path.join(args.out, "cloud_voxel.pt"))
    print(f"  voxel-fused  -> {xyz.shape[0]:,} points")

    per_img = TARGET // len(cams)
    ux, ur = build_cloud_uniform(cams, pnorm, mask, R, dev, per_img)
    torch.save({"xyz": ux.cpu(), "rgb": ur.cpu(), "raw_n": raw_n, "R": R},
               os.path.join(args.out, "cloud_uniform.pt"))
    print(f"  uniform {per_img}/image -> {ux.shape[0]:,} points")

    sx_, sr_, sn_ = build_cloud_surface(cams, pnorm, mask, R, dev,
                                        voxel=args.voxel, cap=TARGET)
    torch.save({"xyz": sx_.cpu(), "rgb": sr_.cpu(), "normal": sn_.cpu(),
                "raw_n": raw_n, "R": R},
               os.path.join(args.out, "cloud_surface.pt"))

    # per-view metric depth + object mask, for depth-guided density control
    dmaps = torch.stack([depth_from_radius(pnorm[v], cams[v].cam_pos.norm(), R)
                         for v in range(len(cams))])
    torch.save({"depth": dmaps.half().cpu(), "mask": mask.cpu(), "R": R},
               os.path.join(args.out, "depthmaps.pt"))
    print(f"  depth maps -> {tuple(dmaps.shape)}")

    hx, hr = build_cloud_hybrid(cams, pnorm, mask, R, dev,
                                voxel=args.voxel, cap=TARGET)
    torch.save({"xyz": hx.cpu(), "rgb": hr.cpu(), "raw_n": raw_n, "R": R},
               os.path.join(args.out, "cloud_hybrid.pt"))
    print(f"  hybrid (saliency + fusion) -> {hx.shape[0]:,} points")

    # sanity: how well does the cloud sit inside the camera sphere?
    r = xyz.norm(dim=1)
    print(f"cloud radius: median {float(r.median()):.2f}, "
          f"p95 {float(r.quantile(0.95)):.2f}  (cameras orbit at 4.03)")


# --------------------------------------------------------------------------
def build_cloud_uniform(cams, pnorm, mask, R, dev, per_image):
    """Take a FIXED number of pixels from every image, uniformly at random.

    This is the "subsample each photo so the total hits the target" strategy:
        per_image = target_total / n_images
    It decouples cloud size from image count, which is the right instinct.
    What it cannot do is notice that two cameras are looking at the same
    surface - every view contributes its full quota regardless of overlap, so
    heavily-observed regions are still over-represented and rarely-seen ones
    under-represented. Voxel fusion allocates by 3D coverage instead.
    """
    H, W = cams[0].H, cams[0].W
    fx, fy, cx, cy = cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32),
                            indexing="ij")
    all_xyz, all_rgb = [], []
    for v, c in enumerate(cams):
        sel = torch.nonzero(mask[v].reshape(-1), as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        take = min(per_image, sel.numel())
        sel = sel[torch.randperm(sel.numel(), device=dev)[:take]]
        z = depth_from_radius(pnorm[v].reshape(-1)[sel], c.cam_pos.norm(), R)
        u, vv = xs.reshape(-1)[sel] + 0.5, ys.reshape(-1)[sel] + 0.5
        pc = torch.stack([(u - cx) * z / fx, (vv - cy) * z / fy, z], -1)
        Rw, t = c.viewmat[:3, :3], c.viewmat[:3, 3]
        all_xyz.append((pc - t) @ Rw)
        all_rgb.append(c.image.reshape(-1, 3)[sel])
    return torch.cat(all_xyz), torch.cat(all_rgb)


# --------------------------------------------------------------------------
def normals_from_depth(z_map, fx, fy, cx, cy, Rw):
    """Surface normal per pixel, straight from the depth map's gradients.

    A depth map is a height field over the image plane, so the surface tangent
    vectors are just the derivatives of the unprojected point with respect to
    u and v; their cross product is the normal. This information is already
    sitting in the depth prediction and is normally discarded - 3DGS then has
    to rediscover surface orientation through gradient descent.

    Returns world-space unit normals [H,W,3], oriented to face the camera.
    """
    H, W = z_map.shape
    dev = z_map.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32),
                            indexing="ij")
    P = torch.stack([(xs + 0.5 - cx) * z_map / fx,
                     (ys + 0.5 - cy) * z_map / fy, z_map], -1)      # [H,W,3]
    du = torch.zeros_like(P); dv = torch.zeros_like(P)
    du[:, 1:-1] = (P[:, 2:] - P[:, :-2]) * 0.5
    dv[1:-1, :] = (P[2:, :] - P[:-2, :]) * 0.5
    n = torch.cross(du, dv, dim=-1)
    n = F.normalize(n, dim=-1, eps=1e-8)
    # camera looks down +z, so a visible surface's normal must have n.z < 0
    n = torch.where(n[..., 2:3] > 0, -n, n)
    return n @ Rw                                    # camera -> world


def build_cloud_surface(cams, pnorm, mask, R, dev, voxel=0.02, cap=50_000):
    """Depth cloud PLUS a surface normal for every point.

    Same geometry as the voxel-fused cloud, but each surviving point also
    carries the local surface orientation, so gaussians can be initialised as
    flat discs lying ON the surface instead of as isotropic blobs.
    """
    H, W = cams[0].H, cams[0].W
    fx, fy, cx, cy = cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32),
                            indexing="ij")
    axyz, argb, anrm = [], [], []
    for v, c in enumerate(cams):
        zmap = depth_from_radius(pnorm[v], c.cam_pos.norm(), R)      # [H,W]
        Rw = c.viewmat[:3, :3]
        nrm = normals_from_depth(zmap, fx, fy, cx, cy, Rw).reshape(-1, 3)
        sel = torch.nonzero(mask[v].reshape(-1), as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        z = zmap.reshape(-1)[sel]
        u, vv = xs.reshape(-1)[sel] + 0.5, ys.reshape(-1)[sel] + 0.5
        pc = torch.stack([(u - cx) * z / fx, (vv - cy) * z / fy, z], -1)
        t = c.viewmat[:3, 3]
        axyz.append((pc - t) @ Rw)
        argb.append(c.image.reshape(-1, 3)[sel])
        anrm.append(nrm[sel])
    xyz, rgb, nrm = torch.cat(axyz), torch.cat(argb), torch.cat(anrm)

    # voxel-fuse, averaging normals within each voxel too
    key = torch.round(xyz / voxel).long(); key = key - key.min(0).values
    mul = torch.tensor([1, key[:, 0].max() + 1,
                        (key[:, 0].max() + 1) * (key[:, 1].max() + 1)], device=dev)
    uniq, inv = torch.unique((key * mul).sum(1), return_inverse=True)
    cnt = torch.zeros(uniq.numel(), device=dev).index_add_(
        0, inv, torch.ones(xyz.shape[0], device=dev))
    sx = torch.zeros(uniq.numel(), 3, device=dev).index_add_(0, inv, xyz)
    sc = torch.zeros(uniq.numel(), 3, device=dev).index_add_(0, inv, rgb)
    sn = torch.zeros(uniq.numel(), 3, device=dev).index_add_(0, inv, nrm)
    xyz_v, rgb_v = sx / cnt[:, None], sc / cnt[:, None]
    nrm_v = F.normalize(sn, dim=-1, eps=1e-8)
    if xyz_v.shape[0] > cap:
        k = torch.randperm(xyz_v.shape[0], device=dev)[:cap]
        xyz_v, rgb_v, nrm_v = xyz_v[k], rgb_v[k], nrm_v[k]
    print(f"  surface cloud -> {xyz_v.shape[0]:,} points with normals")
    return xyz_v, rgb_v, nrm_v


# --------------------------------------------------------------------------
def build_cloud_hybrid(cams, pnorm, mask, R, dev, voxel=0.02, cap=50_000,
                       floor=0.25):
    """Best of both: detail-aware per-image sampling, THEN cross-view fusion.

    Uniform per-image sampling respects nothing about image content - it
    spends as many points on a flat yellow panel as on a row of studs.
    Voxel fusion dedupes across views but is uniform in 3D, so it cannot
    put extra points where the image actually has detail.

    This combines them:
      1. per view, sample pixels with probability proportional to local image
         gradient (plus a uniform floor so flat regions are not abandoned),
         which concentrates points on edges and texture;
      2. fuse all views and voxel-dedupe, which removes the cross-view
         duplication that per-image sampling leaves behind.

    Detail-aware allocation AND no redundancy.
    """
    H, W = cams[0].H, cams[0].W
    fx, fy, cx, cy = cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32),
                            indexing="ij")
    # oversample 4x before fusion so the voxel step has something to choose from
    per_image = int(4 * cap / len(cams))

    all_xyz, all_rgb = [], []
    for v, c in enumerate(cams):
        g = c.image.mean(-1)[None, None]
        gx = (g[..., :, 1:] - g[..., :, :-1]).abs().mean()
        # Sobel-ish gradient magnitude as a saliency map
        dx = F.pad((g[..., :, 1:] - g[..., :, :-1]).abs(), (0, 1))
        dy = F.pad((g[..., 1:, :] - g[..., :-1, :]).abs(), (0, 0, 0, 1))
        sal = (dx + dy)[0, 0].reshape(-1)
        sal = sal / sal.max().clamp(min=1e-8)
        w = (sal + floor) * mask[v].reshape(-1).float()    # object pixels only
        if w.sum() <= 0:
            continue
        take = min(per_image, int((w > 0).sum()))
        sel = torch.multinomial(w, take, replacement=False)

        z = depth_from_radius(pnorm[v].reshape(-1)[sel], c.cam_pos.norm(), R)
        u, vv = xs.reshape(-1)[sel] + 0.5, ys.reshape(-1)[sel] + 0.5
        pc = torch.stack([(u - cx) * z / fx, (vv - cy) * z / fy, z], -1)
        Rw, t = c.viewmat[:3, :3], c.viewmat[:3, 3]
        all_xyz.append((pc - t) @ Rw)
        all_rgb.append(c.image.reshape(-1, 3)[sel])

    xyz, rgb = torch.cat(all_xyz), torch.cat(all_rgb)
    return voxel_fuse(xyz, rgb, dev, voxel, cap)


if __name__ == "__main__":
    main()
