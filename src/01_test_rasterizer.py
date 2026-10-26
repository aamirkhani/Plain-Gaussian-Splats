"""Validate the rasterizer against INDEPENDENT references.

Four things can be wrong, and each gets its own test:

  [A] the 2D covariance math      -> checked against a Monte-Carlo estimate
                                     (sample the 3D gaussian, project the
                                     samples, measure their 2D covariance)
  [B] compositing + depth order   -> checked against a brute-force reference
                                     that loops gaussians in global depth order
  [C] the tile binning            -> checked by forcing "every gaussian in
                                     every tile" and requiring the tiled path
                                     to reproduce the brute-force result
  [D] the backward pass           -> checked against finite differences

A rasterizer that renders a plausible-looking image can still have a wrong
Jacobian, which shows up only as training that mysteriously plateaus. So [D]
matters as much as the rest.
"""

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rasterizer as R

FAIL = []


def check(name, got, ref, tol):
    got, ref = got.detach().double().cpu(), ref.detach().double().cpu()
    err = (got - ref).abs().max().item()
    ok = err <= tol and torch.isfinite(got).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<42} err={err:.3e} (tol={tol:.1e})")
    if not ok:
        FAIL.append(name)
    return ok


def make_scene(n, dev, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    mean3d = (torch.rand(n, 3, generator=g) - 0.5) * 1.2
    log_scale = torch.log(torch.rand(n, 3, generator=g) * 0.10 + 0.04)
    quat = torch.randn(n, 4, generator=g)
    opacity_raw = torch.rand(n, generator=g) * 2.0 - 0.5
    sh = torch.zeros(n, 16, 3)
    sh[:, 0] = (torch.rand(n, 3, generator=g) - 0.5) * 1.5
    sh[:, 1:] = torch.randn(n, 15, 3, generator=g) * 0.05
    t = [x.to(dev).to(dtype) for x in (mean3d, log_scale, quat, opacity_raw, sh)]
    return t


def camera(dev, dtype=torch.float32, dist=3.0):
    """Camera on the +z axis looking back at the origin."""
    viewmat = torch.eye(4, dtype=dtype, device=dev)
    viewmat[2, 3] = dist                    # world-to-camera translation
    cam_pos = torch.tensor([0.0, 0.0, -dist], dtype=dtype, device=dev)
    return viewmat, cam_pos


# --------------------------------------------------------------------------
# [A] 2D covariance vs Monte Carlo
# --------------------------------------------------------------------------
def test_cov2d(dev):
    print("\n[A] 2D covariance  (analytic  vs  Monte-Carlo projection)")
    torch.manual_seed(0)
    n = 6
    mean3d = (torch.rand(n, 3) - 0.5).to(dev)
    log_scale = torch.log(torch.rand(n, 3) * 0.05 + 0.02).to(dev)
    quat = torch.randn(n, 4).to(dev)
    viewmat, _ = camera(dev)
    fx = fy = 300.0
    cx = cy = 64.0

    mean2d, conic, depth, radius, _ = R.project(
        mean3d, log_scale, quat, viewmat, fx, fy, cx, cy)

    # invert the conic to recover the analytic 2D covariance
    a, b, c = conic[:, 0], conic[:, 1], conic[:, 2]
    det = a * c - b * b
    cov_analytic = torch.stack([c / det, -b / det, -b / det, a / det],
                               dim=-1).reshape(-1, 2, 2)
    # the renderer adds BLUR*I as a low-pass filter; remove it to compare
    cov_analytic[:, 0, 0] -= R.BLUR
    cov_analytic[:, 1, 1] -= R.BLUR

    # Monte Carlo: draw from N(mu, Sigma), project, measure 2D covariance
    Sigma = R.covariance_3d(log_scale, quat)
    L = torch.linalg.cholesky(Sigma.double())
    nsamp = 400_000
    eps = torch.randn(n, nsamp, 3, dtype=torch.float64, device=dev)
    pts = mean3d.double()[:, None, :] + eps @ L.transpose(1, 2)

    Rm, t = viewmat[:3, :3].double(), viewmat[:3, 3].double()
    pc = pts @ Rm.T + t
    u = fx * pc[..., 0] / pc[..., 2] + cx
    v = fy * pc[..., 1] / pc[..., 2] + cy
    uv = torch.stack([u, v], dim=-1)
    uv = uv - uv.mean(dim=1, keepdim=True)
    cov_mc = (uv.transpose(1, 2) @ uv) / (nsamp - 1)

    # scale-relative tolerance: these covariances are O(1..100) px^2
    scale = cov_mc.abs().amax(dim=(1, 2)).clamp_min(1e-3)
    check("cov2d (relative)", cov_analytic.double() / scale[:, None, None],
          cov_mc / scale[:, None, None], 2e-2)

    # projected mean must equal plain pinhole math
    pc_mean = mean3d.double() @ Rm.T + t
    man = torch.stack([fx * pc_mean[:, 0] / pc_mean[:, 2] + cx,
                       fy * pc_mean[:, 1] / pc_mean[:, 2] + cy], dim=-1)
    check("projected mean vs pinhole", mean2d.double(), man, 1e-4)


# --------------------------------------------------------------------------
# [B]/[C] compositing vs brute force
# --------------------------------------------------------------------------
def brute_force(mean2d, conic, opacity, color, depth, H, W, bg):
    """Reference: every pixel sees every gaussian, in global depth order."""
    dev = mean2d.device
    order = torch.argsort(depth)
    m, cn = mean2d[order], conic[order]
    op, co = opacity[order], color[order]

    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=dev),
                            torch.arange(W, dtype=torch.float32, device=dev),
                            indexing="ij")
    px = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1) + 0.5  # [P,2]

    dx = px[:, None, 0] - m[None, :, 0]
    dy = px[:, None, 1] - m[None, :, 1]
    q = 0.5 * (cn[None, :, 0] * dx * dx + 2 * cn[None, :, 1] * dx * dy
               + cn[None, :, 2] * dy * dy)
    alpha = (op[None, :] * torch.exp(-q.clamp_min(0.0))).clamp(0.0, R.MAX_ALPHA)

    one_minus = 1.0 - alpha
    T_excl = torch.cumprod(
        torch.cat([torch.ones_like(one_minus[:, :1]), one_minus[:, :-1]], 1), 1)
    w = alpha * T_excl
    rgb = w @ co
    acc = w.sum(1)
    img = rgb + (1 - acc)[:, None] * bg
    return img.reshape(H, W, 3), acc.reshape(H, W)


def all_tiles_bin(mean2d, radius, depth, n_tx, n_ty):
    """Binning that puts EVERY gaussian in EVERY tile (no culling).

    Substituting this makes the tiled renderer mathematically identical to
    the brute-force reference, isolating compositing from culling.
    """
    n = mean2d.shape[0]
    n_tiles = n_tx * n_ty
    dev = mean2d.device
    pair_tile = torch.arange(n_tiles, device=dev).repeat_interleave(n)
    pair_gauss = torch.arange(n, device=dev).repeat(n_tiles)
    o1 = torch.argsort(depth[pair_gauss])
    o2 = torch.argsort(pair_tile[o1], stable=True)
    order = o1[o2]
    return pair_tile[order], pair_gauss[order]


def test_composite(dev):
    print("\n[B/C] compositing + binning  (tiled  vs  brute force)")
    H = W = 64
    fx = fy = 160.0
    cx, cy = W / 2, H / 2
    mean3d, log_scale, quat, opacity_raw, sh = make_scene(25, dev)
    viewmat, cam_pos = camera(dev)

    mean2d, conic, depth, radius, _ = R.project(
        mean3d, log_scale, quat, viewmat, fx, fy, cx, cy)
    opacity = torch.sigmoid(opacity_raw)
    dirs = torch.nn.functional.normalize(mean3d - cam_pos[None], dim=-1)
    color = (R.eval_sh(sh, dirs, 3) + 0.5).clamp_min(0.0)

    ref_img, ref_acc = brute_force(mean2d, conic, opacity, color, depth, H, W, 1.0)

    # [C] exact comparison with culling disabled
    orig = R.bin_to_tiles
    R.bin_to_tiles = all_tiles_bin
    try:
        img, acc, _ = R.render(mean3d, log_scale, quat, opacity_raw, sh,
                               viewmat, cam_pos, H, W, fx, fy, cx, cy,
                               use_checkpoint=False)
    finally:
        R.bin_to_tiles = orig
    check("tiled == brute force (no culling)", img, ref_img, 1e-5)
    check("alpha == brute force (no culling)", acc, ref_acc, 1e-5)

    # [B] with real 3-sigma culling: should be close, differing only by the
    # truncated tails (alpha < opacity*exp(-4.5) ~ 1%)
    img2, acc2, _ = R.render(mean3d, log_scale, quat, opacity_raw, sh,
                             viewmat, cam_pos, H, W, fx, fy, cx, cy,
                             use_checkpoint=False)
    check("tiled (3-sigma cull) ~ brute force", img2, ref_img, 2e-2)

    # chunking must not change the answer
    img3, _, _ = R.render(mean3d, log_scale, quat, opacity_raw, sh,
                          viewmat, cam_pos, H, W, fx, fy, cx, cy,
                          chunk_elems=20_000, use_checkpoint=False)
    check("small chunks == large chunks", img3, img2, 1e-6)

    # checkpointing must not change the forward value either
    img4, _, _ = R.render(mean3d, log_scale, quat, opacity_raw, sh,
                          viewmat, cam_pos, H, W, fx, fy, cx, cy,
                          use_checkpoint=True)
    check("checkpointed == plain forward", img4, img2, 1e-6)


# --------------------------------------------------------------------------
# [D] gradients vs finite differences
# --------------------------------------------------------------------------
def test_gradients(dev):
    print("\n[D] backward pass  (analytic  vs  central finite differences)")
    H = W = 48
    fx = fy = 110.0
    cx, cy = W / 2, H / 2
    # float64 so the finite differences are meaningful
    mean3d, log_scale, quat, opacity_raw, sh = make_scene(
        10, dev, dtype=torch.float64, seed=3)
    viewmat, cam_pos = camera(dev, dtype=torch.float64)
    torch.manual_seed(7)
    target = torch.rand(H, W, 3, dtype=torch.float64, device=dev)

    params = {"mean3d": mean3d, "log_scale": log_scale, "quat": quat,
              "opacity_raw": opacity_raw, "sh": sh}
    for v in params.values():
        v.requires_grad_(True)

    def loss_fn():
        img, _, _ = R.render(params["mean3d"], params["log_scale"],
                             params["quat"], params["opacity_raw"],
                             params["sh"], viewmat, cam_pos, H, W,
                             fx, fy, cx, cy, use_checkpoint=False)
        return ((img - target) ** 2).mean()

    loss = loss_fn()
    loss.backward()
    analytic = {k: v.grad.clone() for k, v in params.items()}

    eps = 1e-6
    for name in params:
        p = params[name]
        flat = p.view(-1)
        # probe a handful of coordinates with the largest analytic gradient,
        # since near-zero entries are dominated by truncation noise
        mag = analytic[name].view(-1).abs()
        probe = torch.topk(mag, min(6, mag.numel())).indices.tolist()

        num, ana = [], []
        for i in probe:
            with torch.no_grad():
                orig = flat[i].item()
                flat[i] = orig + eps
                lp = loss_fn().item()
                flat[i] = orig - eps
                lm = loss_fn().item()
                flat[i] = orig
            num.append((lp - lm) / (2 * eps))
            ana.append(analytic[name].view(-1)[i].item())
        num = torch.tensor(num, dtype=torch.float64)
        ana = torch.tensor(ana, dtype=torch.float64)
        scale = ana.abs().max().clamp_min(1e-8)
        check(f"d loss / d {name}", ana / scale, num / scale, 2e-4)


# --------------------------------------------------------------------------
def test_gpu_cpu_agreement(dev):
    print("\n[E] GPU render == CPU render")
    H = W = 80
    fx = fy = 180.0
    args_cpu = make_scene(40, "cpu")
    vm_c, cp_c = camera("cpu")
    img_c, _, _ = R.render(*args_cpu, vm_c, cp_c, H, W, fx, fy, W / 2, H / 2,
                           use_checkpoint=False)
    args_gpu = [a.to(dev) for a in args_cpu]
    vm_g, cp_g = camera(dev)
    img_g, _, _ = R.render(*args_gpu, vm_g, cp_g, H, W, fx, fy, W / 2, H / 2,
                           use_checkpoint=False)
    check("GPU image == CPU image", img_g, img_c, 1e-4)


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}  "
          f"({torch.cuda.get_device_properties(0).name if dev.type=='cuda' else 'cpu'})")

    test_cov2d(dev)
    test_composite(dev)
    test_gradients(dev)
    test_gpu_cpu_agreement(dev)

    print("\n" + "=" * 72)
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("RESULT: rasterizer forward AND backward verified against "
          "independent references.")
    print("=" * 72)


if __name__ == "__main__":
    main()
