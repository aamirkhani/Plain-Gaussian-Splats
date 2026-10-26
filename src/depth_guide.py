"""Depth-guided density control.

Every experiment so far used the depth prior to decide WHERE TO PUT POINTS at
iteration 0, and every one of them washed out by iteration 2000 because
adaptive density control rebuilds whatever it is not given. So stop using the
prior as an initialiser and start using it as a SUPERVISOR of density control
itself - the heuristic the evidence says is actually binding.

This is also a different mechanism from the published depth work: DNGaussian
and DN-Splatter add depth/normal terms to the LOSS, paying for them on every
gradient step. Here the prior never touches the loss. It is consulted only at
densification time, every ~100 iterations, to answer two questions that
position gradients cannot:

  PRUNE   "is this gaussian floating in empty space in front of the surface?"
          A screen-space gradient cannot tell a floater from real geometry -
          both reduce the loss locally. Depth can: a gaussian consistently
          nearer than the predicted surface, across the views that see it, is
          not on the object.

  INFILL  "is there surface here that the model has not covered at all?"
          Cloning only multiplies gaussians that ALREADY exist, so a region
          with no gaussians has no gradient to clone from and can stay empty
          forever. The prior knows the surface is there and can seed it.

Both are cheap: a projection per gaussian per view, no rendering for prune,
one alpha render per sampled view for infill.
"""

import torch


@torch.no_grad()
def floater_score(gm, cams, depths, masks, tol=0.05):
    """Fraction of observing views in which each gaussian sits IN FRONT of
    the predicted surface.

    Returns (score [N], n_seen [N]). A high score with enough observations
    means "this thing is floating between the camera and the object".
    """
    dev = gm.mean3d.device
    n = gm.n
    front = torch.zeros(n, device=dev)
    seen = torch.zeros(n, device=dev)
    xyz = gm.mean3d.detach()
    H, W = cams[0].H, cams[0].W
    fx, fy, cx, cy = cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy

    for v, c in enumerate(cams):
        R, t = c.viewmat[:3, :3], c.viewmat[:3, 3]
        pc = xyz @ R.T + t
        z = pc[:, 2]
        u = fx * pc[:, 0] / z.clamp(min=1e-4) + cx
        vv = fy * pc[:, 1] / z.clamp(min=1e-4) + cy
        inb = (z > 0.1) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        fl = (vv.long().clamp(0, H - 1) * W + u.long().clamp(0, W - 1))
        on_obj = masks[v].reshape(-1)[fl] & inb        # lands on the object
        z_prior = depths[v].reshape(-1)[fl]
        # "in front" = nearer to the camera than the predicted surface
        front += (on_obj & (z < z_prior - tol)).float()
        seen += on_obj.float()
    return front / seen.clamp(min=1.0), seen


@torch.no_grad()
def find_holes(gm, cams, depths, masks, render_fn, view_ids,
               alpha_thresh=0.4, per_view=400):
    """Points where the prior says SURFACE but the model renders ~nothing.

    Returns (xyz [M,3], rgb [M,3]) ready to be appended as new gaussians.
    """
    dev = gm.mean3d.device
    H, W = cams[0].H, cams[0].W
    fx, fy, cx, cy = cams[0].fx, cams[0].fy, cams[0].cx, cams[0].cy
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32),
                            indexing="ij")
    out_xyz, out_rgb = [], []
    for v in view_ids:
        c = cams[v]
        _, acc, _ = render_fn(gm, c)                   # accumulated opacity
        # object pixel, but the model put almost no density there
        hole = masks[v] & (acc < alpha_thresh)
        idx = torch.nonzero(hole.reshape(-1), as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        if idx.numel() > per_view:
            idx = idx[torch.randperm(idx.numel(), device=dev)[:per_view]]
        z = depths[v].reshape(-1)[idx]
        u, vv = xs.reshape(-1)[idx] + 0.5, ys.reshape(-1)[idx] + 0.5
        pc = torch.stack([(u - cx) * z / fx, (vv - cy) * z / fy, z], -1)
        R, t = c.viewmat[:3, :3], c.viewmat[:3, 3]
        out_xyz.append((pc - t) @ R)
        out_rgb.append(c.image.reshape(-1, 3)[idx])
    if not out_xyz:
        return None, None
    return torch.cat(out_xyz), torch.cat(out_rgb)


@torch.no_grad()
def apply_depth_guidance(gm, cams, depths, masks, render_fn,
                         do_prune=True, do_infill=True,
                         float_thresh=0.6, min_views=4,
                         n_infill_views=6, cap=None):
    """One round of depth-guided density control. Returns a stats dict."""
    import model as M
    info = {"floaters": 0, "infilled": 0}

    if do_prune:
        score, seen = floater_score(gm, cams, depths, masks)
        kill = (score > float_thresh) & (seen >= min_views)
        # never prune so hard the model collapses
        if kill.any() and int((~kill).sum()) > 1000:
            gm._prune_optimizer(~kill)
            gm.grad_accum = gm.grad_accum[~kill]
            gm.grad_denom = gm.grad_denom[~kill]
            gm.max_radius = gm.max_radius[~kill]
            info["floaters"] = int(kill.sum())

    if do_infill and (cap is None or gm.n < cap):
        vids = torch.randperm(len(cams))[:n_infill_views].tolist()
        xyz, rgb = find_holes(gm, cams, depths, masks, render_fn, vids)
        if xyz is not None and xyz.shape[0] > 0:
            if cap is not None:
                room = max(cap - gm.n, 0)
                if xyz.shape[0] > room:
                    xyz, rgb = xyz[:room], rgb[:room]
            if xyz.shape[0] > 0:
                k = xyz.shape[0]
                dev = gm.mean3d.device
                sp = M.knn_spacing(gm.mean3d.detach()).median()
                extra = {
                    "mean3d": xyz,
                    "sh": torch.zeros(k, M.SH_COEFFS, 3, device=dev),
                    "opacity_raw": torch.full((k, 1),
                                              float(M.inverse_sigmoid(
                                                  torch.tensor(0.1))), device=dev),
                    "log_scale": torch.log(sp).expand(k, 3).contiguous(),
                    "quat": torch.zeros(k, 4, device=dev),
                }
                extra["sh"][:, 0] = (rgb - 0.5) / M.C0
                extra["quat"][:, 0] = 1.0
                gm._append_optimizer(extra)
                pad = torch.zeros(k, device=dev)
                gm.grad_accum = torch.cat([gm.grad_accum, pad])
                gm.grad_denom = torch.cat([gm.grad_denom, pad])
                gm.max_radius = torch.cat([gm.max_radius, pad])
                info["infilled"] = k
    return info
