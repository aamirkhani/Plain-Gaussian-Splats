"""The Gaussian cloud: parameters, optimizer, and adaptive density control.

The parameters are stored in a form that keeps them valid without constraints:
    log_scale   -> exp()     keeps scales positive
    quat        -> normalize() keeps rotation a unit quaternion
    opacity_raw -> sigmoid() keeps opacity in [0,1]

ADAPTIVE DENSITY CONTROL is the part that makes 3DGS work, and it is not
gradient descent. Periodically we look at the accumulated magnitude of the
screen-space position gradient of each gaussian; a large value means "this
gaussian is being pulled in conflicting directions and cannot represent the
detail here on its own". Those get:
    CLONED  if they are small  (add geometry where the scene is under-covered)
    SPLIT   if they are large  (subdivide a blob that is straddling detail)
and separately, nearly transparent gaussians are PRUNED.

Because this changes the number of gaussians, the Adam state has to be
surgically re-indexed alongside the parameters - that is what _prune_optimizer
and _append_optimizer do. Getting this wrong silently poisons training with
stale momentum, so it is handled explicitly.
"""

import torch
import torch.nn.functional as F

SH_DEG = 3
SH_COEFFS = (SH_DEG + 1) ** 2      # 16
C0 = 0.28209479177387814           # SH band-0 constant


def knn_spacing(pts, k=4, chunk=2048):
    """Mean distance to the k nearest neighbours, per point.

    Used to size gaussians initialised from a real point cloud: a splat should
    be about as big as the gap to its neighbours, so dense regions get small
    gaussians and sparse ones get large. Chunked because the full N x N
    distance matrix does not fit.
    """
    n = pts.shape[0]
    out = torch.empty(n, device=pts.device)
    for i in range(0, n, chunk):
        d = torch.cdist(pts[i:i + chunk], pts)
        out[i:i + chunk] = d.topk(k + 1, largest=False).values[:, 1:].mean(1)
    return out


def rot_to_quat(R):
    """Rotation matrices [N,3,3] -> unit quaternions [N,4] as (w,x,y,z).

    Shepperd's method: pick whichever of the four formulas has the largest
    denominator, so we never divide by something near zero.
    """
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(m.shape[0], 4, device=m.device)
    c0 = t > 0
    s = torch.sqrt((t[c0] + 1.0).clamp(min=1e-12)) * 2
    q[c0, 0] = 0.25 * s
    q[c0, 1] = (m[c0, 2, 1] - m[c0, 1, 2]) / s
    q[c0, 2] = (m[c0, 0, 2] - m[c0, 2, 0]) / s
    q[c0, 3] = (m[c0, 1, 0] - m[c0, 0, 1]) / s
    rest = ~c0
    if rest.any():
        mm = m[rest]
        d = torch.stack([mm[:, 0, 0], mm[:, 1, 1], mm[:, 2, 2]], 1)
        big = d.argmax(1)
        qr = torch.zeros(mm.shape[0], 4, device=m.device)
        for ax in range(3):
            sel = big == ax
            if not sel.any():
                continue
            a, b, c = ax, (ax + 1) % 3, (ax + 2) % 3
            ms = mm[sel]
            s2 = torch.sqrt((1.0 + ms[:, a, a] - ms[:, b, b]
                             - ms[:, c, c]).clamp(min=1e-12)) * 2
            qr[sel, 0] = (ms[:, c, b] - ms[:, b, c]) / s2
            qr[sel, 1 + a] = 0.25 * s2
            qr[sel, 1 + b] = (ms[:, b, a] + ms[:, a, b]) / s2
            qr[sel, 1 + c] = (ms[:, c, a] + ms[:, a, c]) / s2
        q[rest] = qr
    return F.normalize(q, dim=-1)


def frame_from_normal(n):
    """Build a rotation whose THIRD axis is the surface normal.

    Columns 1 and 2 then span the tangent plane, so shrinking the third scale
    turns the gaussian into a disc lying flat on the surface.
    """
    n = F.normalize(n, dim=-1, eps=1e-8)
    # a helper that is never parallel to n: pick the world axis n is smallest along
    helper = torch.zeros_like(n)
    helper[torch.arange(n.shape[0]), n.abs().argmin(dim=1)] = 1.0
    t1 = F.normalize(torch.cross(helper, n, dim=-1), dim=-1, eps=1e-8)
    t2 = torch.cross(n, t1, dim=-1)
    return torch.stack([t1, t2, n], dim=-1)          # columns = axes


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class GaussianModel:
    def __init__(self, n, extent, device, lr_scale=None, random_scale=1.3,
                 init_xyz=None, init_rgb=None, init_opacity=0.1,
                 init_normal=None, flat=0.25):
        self.device = device
        self.extent = extent
        self.lr_scale = lr_scale if lr_scale is not None else extent
        self.sh_degree_active = 0          # warm up SH bands gradually

        if init_xyz is not None:
            # Initialise from a supplied point cloud (SfM, stereo, or a
            # monocular depth network). Scale comes from local density so a
            # splat is about as big as the gap to its neighbours.
            pts = init_xyz.to(device).float()
            n = pts.shape[0]
            sp = knn_spacing(pts).clamp(min=1e-4)
            log_scale = torch.log(sp)[:, None].repeat(1, 3)
            if init_normal is not None:
                # SURFACE-ALIGNED INIT: squash the axis along the normal so the
                # gaussian starts as a disc lying ON the surface, which is the
                # shape 3DGS converges to anyway. The orientation is free - it
                # comes from the depth map's own gradients.
                log_scale[:, 2] = torch.log(sp * flat)
        else:
            # random init inside a cube; NeRF-synthetic ships no SfM cloud
            pts = (torch.rand(n, 3, device=device) - 0.5) * 2.0 * random_scale
            # initial scale: roughly the mean nearest-neighbour spacing
            spacing = (2.0 * random_scale) / (n ** (1.0 / 3.0))
            log_scale = torch.log(torch.full((n, 3), spacing * 0.5, device=device))

        if init_xyz is not None and init_normal is not None:
            quat = rot_to_quat(frame_from_normal(init_normal.to(device).float()))
        else:
            quat = torch.zeros(n, 4, device=device)
            quat[:, 0] = 1.0

        # Starting opacity. 0.1 (near-transparent) is right for RANDOM init,
        # where most gaussians are in empty space and should fade out and be
        # pruned. Points from a depth network are already on surfaces, so a
        # higher value may let them contribute sooner instead of spending
        # iterations climbing back up.
        opacity_raw = inverse_sigmoid(
            torch.full((n, 1), float(init_opacity), device=device))

        sh = torch.zeros(n, SH_COEFFS, 3, device=device)
        if init_rgb is not None:
            # invert colour = C0*sh0 + 0.5 so the splat starts the pixel's colour
            sh[:, 0] = (init_rgb.to(device).float() - 0.5) / C0
        # else leave at 0 -> mid grey after the +0.5 offset

        self.mean3d = pts.requires_grad_(True)
        self.log_scale = log_scale.requires_grad_(True)
        self.quat = quat.requires_grad_(True)
        self.opacity_raw = opacity_raw.requires_grad_(True)
        self.sh = sh.requires_grad_(True)

        # densification bookkeeping
        self.grad_accum = torch.zeros(n, device=device)
        self.grad_denom = torch.zeros(n, device=device)
        self.max_radius = torch.zeros(n, device=device)

        self._make_optimizer()

    # ------------------------------------------------------------------
    @property
    def n(self):
        return self.mean3d.shape[0]

    def _make_optimizer(self):
        # Position lr scales with scene size; the others are scale-free.
        self.opt = torch.optim.Adam([
            {"params": [self.mean3d], "lr": 1.6e-4 * self.lr_scale, "name": "mean3d"},
            {"params": [self.sh], "lr": 2.5e-3, "name": "sh"},
            {"params": [self.opacity_raw], "lr": 5e-2, "name": "opacity_raw"},
            {"params": [self.log_scale], "lr": 5e-3, "name": "log_scale"},
            {"params": [self.quat], "lr": 1e-3, "name": "quat"},
        ], eps=1e-15)
        self._pos_lr_init = 1.6e-4 * self.lr_scale
        self._pos_lr_final = 1.6e-6 * self.lr_scale

    def update_lr(self, it, total):
        """Exponential decay on the position lr only (as in the paper)."""
        t = min(max(it / max(total, 1), 0.0), 1.0)
        lr = self._pos_lr_init * (self._pos_lr_final / self._pos_lr_init) ** t
        self.opt.param_groups[0]["lr"] = lr
        return lr

    def oneup_sh_degree(self):
        if self.sh_degree_active < SH_DEG:
            self.sh_degree_active += 1

    # ------------------------------------------------------------------
    # optimizer-state surgery
    # ------------------------------------------------------------------
    def _tensors(self):
        return ["mean3d", "sh", "opacity_raw", "log_scale", "quat"]

    def _prune_optimizer(self, mask):
        """Keep only entries where mask is True, in params AND Adam moments."""
        for group in self.opt.param_groups:
            old = group["params"][0]
            state = self.opt.state.get(old, None)
            new = old.detach()[mask].clone().requires_grad_(True)
            if state is not None:
                del self.opt.state[old]
                state["exp_avg"] = state["exp_avg"][mask].clone()
                state["exp_avg_sq"] = state["exp_avg_sq"][mask].clone()
                self.opt.state[new] = state
            group["params"][0] = new
            setattr(self, group["name"], new)

    def _append_optimizer(self, extra):
        """Concatenate new gaussians, with ZERO Adam moments for them."""
        for group in self.opt.param_groups:
            name = group["name"]
            old = group["params"][0]
            add = extra[name]
            state = self.opt.state.get(old, None)
            new = torch.cat([old.detach(), add], dim=0).requires_grad_(True)
            if state is not None:
                del self.opt.state[old]
                state["exp_avg"] = torch.cat(
                    [state["exp_avg"], torch.zeros_like(add)], dim=0)
                state["exp_avg_sq"] = torch.cat(
                    [state["exp_avg_sq"], torch.zeros_like(add)], dim=0)
                self.opt.state[new] = state
            group["params"][0] = new
            setattr(self, name, new)

    # ------------------------------------------------------------------
    def record_stats(self, stats, W):
        """Accumulate the densification signal after a backward pass."""
        g = stats["mean2d"].grad
        if g is None:
            return
        vis = stats["visible"]
        # convert pixel-space grad to NDC-space so the threshold is
        # resolution independent (u = u_ndc * W/2 + cx)
        gn = g[vis, :2].norm(dim=-1) * (0.5 * W)
        self.grad_accum[vis] += gn
        self.grad_denom[vis] += 1.0
        self.max_radius[vis] = torch.maximum(
            self.max_radius[vis], stats["radius"].detach()[vis])

    def _reset_stats(self):
        n = self.n
        d = self.device
        self.grad_accum = torch.zeros(n, device=d)
        self.grad_denom = torch.zeros(n, device=d)
        self.max_radius = torch.zeros(n, device=d)

    # ------------------------------------------------------------------
    def densify_and_prune(self, grad_thresh, min_opacity, max_screen,
                          percent_dense=0.01, cap=None):
        grads = self.grad_accum / self.grad_denom.clamp_min(1.0)
        big_grad = grads > grad_thresh

        scales = torch.exp(self.log_scale)
        small = scales.max(dim=1).values <= percent_dense * self.extent

        n_before = self.n
        headroom = None if cap is None else max(cap - n_before, 0)

        # ---- CLONE: duplicate small, high-gradient gaussians ----------
        clone_mask = big_grad & small
        if headroom is not None and int(clone_mask.sum()) > headroom:
            clone_mask = self._trim(clone_mask, grads, headroom)
        extra = {n: getattr(self, n).detach()[clone_mask].clone()
                 for n in self._tensors()}
        n_clone = int(clone_mask.sum())

        # ---- SPLIT: subdivide large, high-gradient gaussians ----------
        if headroom is not None:
            headroom = max(headroom - n_clone, 0)
        split_mask = big_grad & (~small)
        if headroom is not None and int(split_mask.sum()) > headroom:
            split_mask = self._trim(split_mask, grads, headroom)
        n_split = int(split_mask.sum())

        if n_split > 0:
            s = scales[split_mask]
            # sample the child positions from the parent's own distribution
            eps = torch.randn(n_split, 3, device=self.device) * s
            Rm = _quat_to_rot(self.quat.detach()[split_mask])
            offset = torch.bmm(Rm, eps[:, :, None]).squeeze(-1)
            extra["mean3d"] = torch.cat(
                [extra["mean3d"], self.mean3d.detach()[split_mask] + offset], 0)
            extra["log_scale"] = torch.cat(
                [extra["log_scale"],
                 torch.log(s / 1.6)], 0)
            for nm in ("sh", "opacity_raw", "quat"):
                extra[nm] = torch.cat(
                    [extra[nm], getattr(self, nm).detach()[split_mask].clone()], 0)

        n_added = n_clone + n_split
        if n_added > 0:
            self._append_optimizer(extra)
            # shrink the split PARENTS so the pair does not double the density
            with torch.no_grad():
                pad = torch.zeros(n_added, dtype=torch.bool, device=self.device)
                parent = torch.cat([split_mask, pad])
                self.log_scale[parent] = torch.log(
                    torch.exp(self.log_scale[parent]) / 1.6)

        # ---- PRUNE ----------------------------------------------------
        # max_radius was recorded for the PRE-append gaussians, so pad it;
        # freshly created gaussians have no screen-size history yet. Note the
        # stats must be read BEFORE resetting them.
        radii = torch.cat([self.max_radius,
                           torch.zeros(n_added, device=self.device)])
        opa = torch.sigmoid(self.opacity_raw).reshape(-1)
        kill = opa < min_opacity
        if max_screen is not None:
            kill = kill | (radii > max_screen)
        kill = kill | (torch.exp(self.log_scale).max(dim=1).values
                       > 0.5 * self.extent)
        n_killed = int(kill.sum())
        if n_killed > 0 and int((~kill).sum()) > 16:
            self._prune_optimizer(~kill)
        else:
            n_killed = 0
        self._reset_stats()

        return {"before": n_before, "cloned": n_clone, "split": n_split,
                "pruned": n_killed, "after": self.n}

    @staticmethod
    def _trim(mask, score, k):
        """Keep only the top-k entries of mask, ranked by score."""
        idx = torch.nonzero(mask, as_tuple=True)[0]
        if k <= 0:
            return torch.zeros_like(mask)
        keep = idx[torch.topk(score[idx], k).indices]
        out = torch.zeros_like(mask)
        out[keep] = True
        return out

    def reset_opacity(self, value=0.01):
        """Knock opacities down; lets floaters die and real surfaces recover."""
        with torch.no_grad():
            new = torch.full_like(self.opacity_raw, float(inverse_sigmoid(
                torch.tensor(value))))
            cur = self.opacity_raw.detach()
            self.opacity_raw.copy_(torch.minimum(cur, new))

    # ------------------------------------------------------------------
    def state(self):
        return {n: getattr(self, n).detach().cpu() for n in self._tensors()} | {
            "extent": self.extent, "sh_degree_active": self.sh_degree_active}


def _quat_to_rot(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
