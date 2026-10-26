"""A differentiable 3D Gaussian Splatting rasterizer in pure PyTorch.

No CUDA, no HIP, no custom kernels - so it runs on the AMD 890M iGPU (or CPU).
Autograd handles the backward pass; we only have to get the forward right.

THE PIPELINE (this is the whole algorithm)
------------------------------------------
Each gaussian is defined in world space by
    mean      mu  in R^3
    scale     s   in R^3   (stored as log s, so it stays positive)
    rotation  q   a unit quaternion
    opacity   o   in [0,1]  (stored as logit, so sigmoid keeps it in range)
    colour    SH coefficients (view-dependent)

To draw it we:
  1. PROJECT the mean to the image plane with a pinhole camera.
  2. Build the 3D covariance  Sigma = R S S^T R^T  from (q, s).
  3. Push it through the projection's Jacobian to get a 2D covariance
     Sigma' = J W Sigma W^T J^T.  A gaussian stays a gaussian under an
     affine map, which is the entire reason this method is fast.
  4. BIN each gaussian into the 16x16 pixel tiles its 3-sigma ellipse touches.
  5. SORT the gaussians inside every tile front-to-back by depth.
  6. COMPOSITE them per pixel with the standard over operator:
         C = sum_i  c_i * alpha_i * T_i,     T_i = prod_{j<i} (1 - alpha_j)
     where alpha_i = o_i * exp(-0.5 * d^T Sigma'^-1 d) for pixel offset d.

MEMORY (the binding constraint here: ~12 GB shared with the CPU)
----------------------------------------------------------------
Step 6 wants a tensor of shape [tiles, pixels_per_tile, gaussians_per_tile],
which explodes. Two mitigations:
  * process tiles in CHUNKS sized so that product stays near `chunk_elems`;
  * wrap each chunk in gradient CHECKPOINTING, so the forward activations are
    recomputed during backward instead of all being held at once.
Together these make peak memory independent of scene complexity.

Also note the colour accumulation is done as a matmul
    [T, P, K] @ [T, K, 3] -> [T, P, 3]
rather than by broadcasting colours to [T, P, K, 3]. Same result, one less
large tensor.
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Tile edge in pixels. Compositing work per gaussian is (2*radius + TILE)^2,
# so smaller tiles waste fewer pixels per splat but create more binning pairs.
# Measured on the 890M (see 02_bench_tile.py): 8 beats 16 by ~1.33x and beats
# 4 by ~1.15x, at both 200px and 400px.
TILE = 8
NEAR = 0.2           # near plane; gaussians closer than this are culled
MAX_ALPHA = 0.99     # clamp so log(1-alpha) stays finite
BLUR = 0.3           # low-pass term added to the 2D covariance diagonal


# --------------------------------------------------------------------------
# spherical harmonics (view-dependent colour)
# --------------------------------------------------------------------------
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435)


def eval_sh(sh, dirs, degree):
    """Evaluate SH colour.

    sh    : [N, (degree+1)^2, 3] coefficients
    dirs  : [N, 3] unit vectors from camera to gaussian
    returns [N, 3] linear RGB (still needs +0.5 offset / clamping)
    """
    out = C0 * sh[:, 0]
    if degree >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        out = out - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
        if degree >= 2:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            out = (out + C2[0] * xy * sh[:, 4] + C2[1] * yz * sh[:, 5]
                   + C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
                   + C2[3] * xz * sh[:, 7] + C2[4] * (xx - yy) * sh[:, 8])
            if degree >= 3:
                out = (out + C3[0] * y * (3 * xx - yy) * sh[:, 9]
                       + C3[1] * xy * z * sh[:, 10]
                       + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                       + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                       + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
                       + C3[5] * z * (xx - yy) * sh[:, 14]
                       + C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return out


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def quat_to_rot(q):
    """Unit quaternion [N,4] (w,x,y,z) -> rotation matrix [N,3,3]."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def covariance_3d(log_scale, quat):
    """Sigma = R S S^T R^T, built from log-scales and a quaternion."""
    s = torch.exp(log_scale)                       # [N,3]
    R = quat_to_rot(quat)                          # [N,3,3]
    RS = R * s[:, None, :]                         # R @ diag(s)
    return RS @ RS.transpose(1, 2)


def project(mean3d, log_scale, quat, viewmat, fx, fy, cx, cy):
    """World-space gaussians -> screen-space 2D gaussians.

    viewmat : [4,4] world-to-camera
    returns mean2d [N,2], conic [N,3] (inverse 2D covariance a,b,c),
            depth [N], radius [N], p_cam [N,3]
    """
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    p_cam = mean3d @ R.T + t                       # [N,3]
    z = p_cam[:, 2].clamp_min(1e-6)

    # pinhole projection
    mean2d = torch.stack([fx * p_cam[:, 0] / z + cx,
                          fy * p_cam[:, 1] / z + cy], dim=-1)

    # Jacobian of the projection at this point: d(u,v)/d(x,y,z) in camera space
    zeros = torch.zeros_like(z)
    J = torch.stack([
        fx / z, zeros, -fx * p_cam[:, 0] / (z * z),
        zeros, fy / z, -fy * p_cam[:, 1] / (z * z),
    ], dim=-1).reshape(-1, 2, 3)

    Sigma = covariance_3d(log_scale, quat)         # [N,3,3]
    W = R[None].expand(Sigma.shape[0], 3, 3)
    T = J @ W                                      # [N,2,3]
    cov2d = T @ Sigma @ T.transpose(1, 2)          # [N,2,2]

    # low-pass filter: guarantees every gaussian covers at least ~a pixel,
    # otherwise sub-pixel splats alias badly and get no gradient
    a = cov2d[:, 0, 0] + BLUR
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + BLUR

    det = (a * c - b * b).clamp_min(1e-8)
    conic = torch.stack([c / det, -b / det, a / det], dim=-1)

    # 3-sigma extent from the larger eigenvalue
    mid = 0.5 * (a + c)
    tmp = (mid * mid - det).clamp_min(1e-8).sqrt()
    radius = 3.0 * torch.sqrt(mid + tmp)

    return mean2d, conic, p_cam[:, 2], radius, p_cam


# --------------------------------------------------------------------------
# tile binning
# --------------------------------------------------------------------------
def bin_to_tiles(mean2d, radius, depth, n_tx, n_ty):
    """Expand each gaussian into (tile, gaussian) pairs it might cover.

    Returns pair_tile [M], pair_gauss [M] with pairs grouped by tile and
    sorted front-to-back inside each tile.
    """
    # integer tile bounding box of the 3-sigma ellipse
    t0x = ((mean2d[:, 0] - radius) / TILE).floor().clamp(0, n_tx).long()
    t1x = ((mean2d[:, 0] + radius) / TILE).ceil().clamp(0, n_tx).long()
    t0y = ((mean2d[:, 1] - radius) / TILE).floor().clamp(0, n_ty).long()
    t1y = ((mean2d[:, 1] + radius) / TILE).ceil().clamp(0, n_ty).long()

    w = (t1x - t0x).clamp_min(0)
    h = (t1y - t0y).clamp_min(0)
    count = w * h                                  # tiles touched per gaussian

    keep = count > 0
    if not keep.any():
        empty = torch.zeros(0, dtype=torch.long, device=mean2d.device)
        return empty, empty

    idx = torch.nonzero(keep, as_tuple=True)[0]
    count, w, t0x, t0y = count[keep], w[keep], t0x[keep], t0y[keep]

    # one entry per (gaussian, tile) pair
    pair_gauss_local = torch.repeat_interleave(
        torch.arange(idx.numel(), device=idx.device), count)
    # position of each pair within its gaussian's bounding box
    start = torch.cumsum(count, 0) - count
    k = torch.arange(int(count.sum()), device=idx.device) - start[pair_gauss_local]
    ty = t0y[pair_gauss_local] + k // w[pair_gauss_local]
    tx = t0x[pair_gauss_local] + k % w[pair_gauss_local]

    pair_tile = ty * n_tx + tx
    pair_gauss = idx[pair_gauss_local]

    # group by tile, depth-ordered within each tile:
    # sort by depth first, then a STABLE sort by tile preserves that order
    o1 = torch.argsort(depth[pair_gauss])
    o2 = torch.argsort(pair_tile[o1], stable=True)
    order = o1[o2]
    return pair_tile[order], pair_gauss[order]


# --------------------------------------------------------------------------
# compositing
# --------------------------------------------------------------------------
def _alpha_pointwise(m, cn, op, valid, pix_xy):
    """The purely pointwise part: pixel offsets -> per-gaussian alpha.

    Split out because this is the piece worth FUSING. It allocates the four
    largest intermediates (dx, dy, q, alpha), each [T,P,K], so fusing them
    keeps the values in registers instead of round-tripping through DRAM -
    and this kernel is bandwidth bound (measured: ~46 GB/s ceiling, arithmetic
    intensity ~0.25 FLOP/byte against a balance point of ~129).

    The transmittance scan is deliberately NOT included here: compiling
    cumprod and its backward made Triton compile times explode (>20 min,
    never completed), while fusing just this prefix compiles in ~18 s and
    measured 1.86x end-to-end.
    """
    dx = pix_xy[:, :, None, 0] - m[:, None, :, 0]  # [T,P,K]
    dy = pix_xy[:, :, None, 1] - m[:, None, :, 1]

    # q = 0.5 * d^T Sigma'^-1 d
    q = 0.5 * (cn[:, None, :, 0] * dx * dx
               + 2.0 * cn[:, None, :, 1] * dx * dy
               + cn[:, None, :, 2] * dy * dy)

    alpha = (op[:, None, :] * torch.exp(-q.clamp_min(0.0))).clamp(0.0, MAX_ALPHA)
    return alpha * valid[:, None, :]               # kill padding


# indirection kept so fusion can be A/B tested without touching call sites
_alpha_impl = _alpha_pointwise


def enable_fusion():
    """Compile the pointwise prefix with Triton. Returns True on success.

    Measured on the 890M: 1.86x end-to-end, image error 3.6e-07 and relative
    gradient error 5.5e-04 vs eager fp32 - i.e. fp32 round-off, not a change
    in the training signal. (By contrast bf16 was only 1.2-1.4x and carried
    41% relative gradient error, so it was rejected.)
    """
    global _alpha_impl
    try:
        _alpha_impl = torch.compile(_alpha_pointwise, dynamic=True)
        return True
    except Exception:
        _alpha_impl = _alpha_pointwise
        return False


def disable_fusion():
    global _alpha_impl
    _alpha_impl = _alpha_pointwise


def _composite_chunk(mean2d, conic, opacity, color, gauss_idx, valid,
                     pix_xy):
    """Alpha-composite one chunk of tiles.

    gauss_idx : [T, K] long  - which gaussian, padded
    valid     : [T, K] bool  - padding mask
    pix_xy    : [T, P, 2]    - pixel centres for these tiles
    returns colour [T, P, 3] and accumulated alpha [T, P]
    """
    co = color[gauss_idx]                          # [T,K,3]
    alpha = _alpha_impl(mean2d[gauss_idx], conic[gauss_idx],
                        opacity[gauss_idx], valid, pix_xy)

    # transmittance in front of each gaussian: exclusive cumulative product
    one_minus = 1.0 - alpha
    T_excl = torch.cumprod(
        torch.cat([torch.ones_like(one_minus[:, :, :1]), one_minus[:, :, :-1]],
                  dim=2), dim=2)

    weight = alpha * T_excl                        # [T,P,K]
    rgb = torch.bmm(weight, co)                    # [T,P,3]  <- avoids [T,P,K,3]
    acc = weight.sum(dim=2)                        # [T,P]
    return rgb, acc


def render(mean3d, log_scale, quat, opacity_raw, sh, viewmat, cam_pos,
           H, W, fx, fy, cx, cy, sh_degree=3, bg=1.0,
           chunk_elems=8_000_000, use_checkpoint=True):
    """Render one image. Differentiable w.r.t. all gaussian parameters.

    Returns image [H,W,3], alpha [H,W], and a dict of per-gaussian stats
    used later for densification.
    """
    dev, dtype = mean3d.device, mean3d.dtype
    n_tx = math.ceil(W / TILE)
    n_ty = math.ceil(H / TILE)
    n_tiles = n_tx * n_ty
    Hp, Wp = n_ty * TILE, n_tx * TILE              # padded canvas

    mean2d, conic, depth, radius, p_cam = project(
        mean3d, log_scale, quat, viewmat, fx, fy, cx, cy)

    # frustum cull: behind the near plane, or fully outside the padded canvas
    visible = (depth > NEAR) \
        & (mean2d[:, 0] + radius > 0) & (mean2d[:, 0] - radius < Wp) \
        & (mean2d[:, 1] + radius > 0) & (mean2d[:, 1] - radius < Hp)

    # accept [N] or [N,1]; downstream broadcasting needs [N]
    opacity = torch.sigmoid(opacity_raw).reshape(-1)
    dirs = F.normalize(mean3d - cam_pos[None], dim=-1)
    color = (eval_sh(sh, dirs, sh_degree) + 0.5).clamp_min(0.0)

    # retain mean2d grad: its magnitude is the densification signal
    if mean2d.requires_grad:
        mean2d.retain_grad()

    stats = {"mean2d": mean2d, "radius": radius, "visible": visible,
             "depth": depth}

    if not bool(visible.any()):
        img = torch.full((H, W, 3), bg, device=dev, dtype=dtype)
        return img, torch.zeros(H, W, device=dev, dtype=dtype), stats

    # mask out culled gaussians by forcing radius 0 so they bin to nothing
    pair_tile, pair_gauss = bin_to_tiles(
        mean2d.detach(), torch.where(visible, radius.detach(),
                                     torch.zeros_like(radius)),
        depth.detach(), n_tx, n_ty)

    if pair_gauss.numel() == 0:
        img = torch.full((H, W, 3), bg, device=dev, dtype=dtype)
        return img, torch.zeros(H, W, device=dev, dtype=dtype), stats

    counts = torch.bincount(pair_tile, minlength=n_tiles)
    offsets = torch.cumsum(counts, 0) - counts
    stats["n_pairs"] = int(pair_gauss.numel())

    P = TILE * TILE
    # pixel centres inside a tile, laid out row-major
    dy_t, dx_t = torch.meshgrid(
        torch.arange(TILE, device=dev, dtype=dtype),
        torch.arange(TILE, device=dev, dtype=dtype), indexing="ij")
    local = torch.stack([dx_t.reshape(-1), dy_t.reshape(-1)], dim=-1) + 0.5

    tile_ids = torch.arange(n_tiles, device=dev)
    tile_origin = torch.stack([(tile_ids % n_tx) * TILE,
                               (tile_ids // n_tx) * TILE], dim=-1).to(dtype)

    # --- walk the tiles in chunks ---------------------------------------
    # Two things matter for speed here:
    #  1. plan the chunks from a CPU copy of `counts`. Reading counts[i].item()
    #     per tile forces a GPU sync per tile (~n_tiles syncs per render),
    #     which dominated everything else.
    #  2. process tiles in order of INCREASING gaussian count. Each chunk pads
    #     to its own max, so grouping similar tiles together removes most of
    #     the padding waste a busy tile would otherwise impose on quiet ones.
    counts_cpu = counts.cpu()
    nonempty = torch.nonzero(counts_cpu > 0, as_tuple=True)[0]
    order = nonempty[torch.argsort(counts_cpu[nonempty])]        # ascending
    order_cpu = order.tolist()
    kcpu = counts_cpu.tolist()

    out_rgb, out_acc, out_tid = [], [], []
    order_dev = order.to(dev)
    i = 0
    n_live = len(order_cpu)
    while i < n_live:
        # ascending order => the last tile in the chunk sets kmax
        j = i + 1
        while j < n_live and (j + 1 - i) * P * kcpu[order_cpu[j]] <= chunk_elems:
            j += 1
        kmax = max(kcpu[order_cpu[j - 1]], 1)
        tsel = order_dev[i:j]                                    # [T]

        ar = torch.arange(kmax, device=dev)
        cnt = counts[tsel]
        valid = ar[None, :] < cnt[:, None]                       # [T,kmax]
        flat = (offsets[tsel][:, None] + ar[None, :]).clamp(
            max=pair_gauss.numel() - 1)
        gidx = pair_gauss[flat] * valid                          # [T,kmax]

        pix = tile_origin[tsel][:, None, :] + local[None]        # [T,P,2]

        args = (mean2d, conic, opacity, color, gidx, valid.float(), pix)
        if use_checkpoint and torch.is_grad_enabled():
            rgb, acc = checkpoint(_composite_chunk, *args, use_reentrant=False)
        else:
            rgb, acc = _composite_chunk(*args)

        out_rgb.append(rgb)
        out_acc.append(acc)
        out_tid.append(tsel)
        i = j

    # tiles were visited out of order, so scatter them back by tile id.
    # index_add on a fresh zeros tensor is differentiable w.r.t. the source,
    # and the tile ids are unique so it acts as a copy.
    tid = torch.cat(out_tid)
    rgb = torch.zeros(n_tiles, P, 3, device=dev, dtype=dtype).index_add(
        0, tid, torch.cat(out_rgb, dim=0))
    acc = torch.zeros(n_tiles, P, device=dev, dtype=dtype).index_add(
        0, tid, torch.cat(out_acc, dim=0))
    rgb = rgb.reshape(n_ty, n_tx, TILE, TILE, 3)
    acc = acc.reshape(n_ty, n_tx, TILE, TILE)

    # [ty, tx, dy, dx] -> [ty, dy, tx, dx] -> full image
    rgb = rgb.permute(0, 2, 1, 3, 4).reshape(Hp, Wp, 3)[:H, :W]
    acc = acc.permute(0, 2, 1, 3).reshape(Hp, Wp)[:H, :W]

    img = rgb + (1.0 - acc)[..., None] * bg        # composite over background
    return img, acc, stats
