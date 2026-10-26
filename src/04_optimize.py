"""Test whether the compositing kernel can be made faster on the 890M.

The roofline says this kernel is memory-bandwidth bound, not ALU bound:
~20 FLOP per ~80 bytes of traffic is an arithmetic intensity of ~0.25
FLOP/byte, while the machine balance is ~49. So the levers worth testing are
the ones that REDUCE TRAFFIC, not the ones that add arithmetic throughput:

  compile : fuse the elementwise chain with Triton so dx/dy/q/alpha stay in
            registers instead of round-tripping through RAM
  bf16    : halve the bytes per element
  both

Every variant is checked for CORRECTNESS against the eager fp32 reference -
image AND gradients - because a fused or reduced-precision kernel is exactly
the kind of change that silently breaks the backward pass.

Run AFTER training finishes; contention invalidates the numbers.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import model as M
import rasterizer as R

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EAGER = R._composite_chunk


# --------------------------------------------------------------------------
def measure_bandwidth():
    """Actual achievable DRAM bandwidth - the ceiling this kernel runs into."""
    n = 64 * 1024 * 1024                      # 256 MiB of fp32
    a = torch.randn(n, device="cuda")
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    t = time.perf_counter()
    reps = 20
    for _ in range(reps):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / reps
    gb = 2 * n * 4 / 1024 ** 3                # read + write
    copy_bw = gb / dt

    # a triad-style op (read 2, write 1) is closer to our access pattern
    c = torch.randn(n, device="cuda")
    for _ in range(3):
        torch.add(a, c, out=b)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        torch.add(a, c, out=b)
    torch.cuda.synchronize()
    dt2 = (time.perf_counter() - t) / reps
    triad_bw = 3 * n * 4 / 1024 ** 3 / dt2
    del a, b, c
    torch.cuda.empty_cache()
    return copy_bw, triad_bw


# --------------------------------------------------------------------------
def composite_bf16(mean2d, conic, opacity, color, gauss_idx, valid, pix):
    """Elementwise chain in bf16; the transmittance scan stays fp32.

    cumprod multiplies up to K terms together - in bf16 (8 mantissa bits)
    that error compounds along the ray, so the scan is kept in fp32 while the
    bulk elementwise traffic is halved.
    """
    h = torch.bfloat16
    m = mean2d[gauss_idx]
    cn = conic[gauss_idx].to(h)
    op = opacity[gauss_idx].to(h)
    co = color[gauss_idx]
    px = pix

    # CRITICAL: do the subtraction in fp32. px and m are ~400 px in magnitude
    # while dx is ~1 px, so subtracting in bf16 (8 mantissa bits, +-1.6 px of
    # error at that magnitude) is catastrophic cancellation - it destroys the
    # very quantity we need. Cast AFTER differencing, when the value is small.
    dx = (px[:, :, None, 0] - m[:, None, :, 0]).to(h)
    dy = (px[:, :, None, 1] - m[:, None, :, 1]).to(h)
    q = 0.5 * (cn[:, None, :, 0] * dx * dx
               + 2.0 * cn[:, None, :, 1] * dx * dy
               + cn[:, None, :, 2] * dy * dy)
    alpha = (op[:, None, :] * torch.exp(-q.clamp_min(0.0))).clamp(0.0, R.MAX_ALPHA)
    alpha = (alpha * valid[:, None, :].to(h)).float()

    one_minus = 1.0 - alpha
    T_excl = torch.cumprod(
        torch.cat([torch.ones_like(one_minus[:, :, :1]), one_minus[:, :, :-1]],
                  dim=2), dim=2)
    weight = alpha * T_excl
    return torch.bmm(weight, co), weight.sum(dim=2)


def composite_bf16_full(mean2d, conic, opacity, color, gauss_idx, valid, pix):
    """Everything in bf16, including the scan - maximum traffic reduction,
    and the variant most likely to lose accuracy. Measured, not assumed."""
    h = torch.bfloat16
    m = mean2d[gauss_idx]
    cn = conic[gauss_idx].to(h)
    op = opacity[gauss_idx].to(h)
    co = color[gauss_idx].to(h)
    px = pix

    # see composite_bf16: the difference must be formed in fp32
    dx = (px[:, :, None, 0] - m[:, None, :, 0]).to(h)
    dy = (px[:, :, None, 1] - m[:, None, :, 1]).to(h)
    q = 0.5 * (cn[:, None, :, 0] * dx * dx
               + 2.0 * cn[:, None, :, 1] * dx * dy
               + cn[:, None, :, 2] * dy * dy)
    alpha = (op[:, None, :] * torch.exp(-q.clamp_min(0.0))).clamp(0.0, R.MAX_ALPHA)
    alpha = alpha * valid[:, None, :].to(h)

    one_minus = 1.0 - alpha
    T_excl = torch.cumprod(
        torch.cat([torch.ones_like(one_minus[:, :, :1]), one_minus[:, :, :-1]],
                  dim=2), dim=2)
    weight = alpha * T_excl
    return torch.bmm(weight, co).float(), weight.sum(dim=2).float()


# --------------------------------------------------------------------------
def pad_pow2(fn):
    """Round K up to a power of two before calling fn.

    Keeps the number of distinct shapes small (so a compiled kernel actually
    amortises) at the cost of some wasted lanes. Correctness is unaffected
    because padded slots are marked invalid.
    """
    def wrapper(mean2d, conic, opacity, color, gauss_idx, valid, pix):
        K = gauss_idx.shape[1]
        K2 = 1 << max(K - 1, 0).bit_length()
        if K2 != K:
            pad = K2 - K
            gauss_idx = torch.nn.functional.pad(gauss_idx, (0, pad))
            valid = torch.nn.functional.pad(valid, (0, pad))
        return fn(mean2d, conic, opacity, color, gauss_idx, valid, pix)
    return wrapper


def run_variant(name, fn, gm, cam, chunk, reps=4, ref=None):
    """Time one variant and check it against the eager fp32 reference."""
    R._composite_chunk = fn
    try:
        # warmup (also triggers compilation)
        t_warm = time.perf_counter()
        for _ in range(2):
            img, _ = _fwd_bwd(gm, cam, chunk)
        torch.cuda.synchronize()
        warm = time.perf_counter() - t_warm

        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(reps):
            img, grads = _fwd_bwd(gm, cam, chunk)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / reps
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3

        if ref is None:
            err_i = err_g = 0.0
        else:
            err_i = (img - ref[0]).abs().max().item()
            err_g = max((g - r).abs().max().item()
                        / max(r.abs().max().item(), 1e-12)
                        for g, r in zip(grads, ref[1]))
        return {"name": name, "s": dt, "warm": warm, "peak": peak,
                "err_img": err_i, "err_grad": err_g,
                "ref": (img.detach().clone(), [g.clone() for g in grads])}
    finally:
        R._composite_chunk = EAGER


def _fwd_bwd(gm, cam, chunk):
    gm.opt.zero_grad(set_to_none=True)
    img, _, _ = R.render(gm.mean3d, gm.log_scale, gm.quat, gm.opacity_raw,
                         gm.sh, cam.viewmat, cam.cam_pos, cam.H, cam.W,
                         cam.fx, cam.fy, cam.cx, cam.cy,
                         sh_degree=gm.sh_degree_active, bg=1.0,
                         chunk_elems=chunk, use_checkpoint=True)
    ((img - cam.image).abs().mean()).backward()
    grads = [gm.mean3d.grad.detach().clone(),
             gm.opacity_raw.grad.detach().clone(),
             gm.log_scale.grad.detach().clone()]
    return img.detach(), grads


# --------------------------------------------------------------------------
def main():
    args_compile = "--no-compile" not in sys.argv
    dev = torch.device("cuda:0")
    chunk = 16_000_000

    print("=" * 78)
    print("MEASURED MEMORY BANDWIDTH (the ceiling for this kernel)")
    copy_bw, triad_bw = measure_bandwidth()
    print(f"  device-to-device copy : {copy_bw:6.1f} GB/s")
    print(f"  triad (2 read 1 write): {triad_bw:6.1f} GB/s")
    peak_flops = 1024 * 2 * 2.9e9 / 1e12
    print(f"  implied balance point : {peak_flops*1000/triad_bw:5.1f} FLOP/byte "
          f"(fp32 peak ~{peak_flops:.1f} TFLOP/s)")

    cams = D.load_blender(os.path.join(HERE, "data/lego"), "train", 400, dev,
                          limit=1)
    cam = cams[0]

    # use the TRAINED cloud: realistic gaussian count and tile occupancy
    ckpt = os.path.join(HERE, "logs/run400/ckpt_last.pt")
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        st = ck["state"]
        gm = M.GaussianModel(st["mean3d"].shape[0], st["extent"], dev)
        with torch.no_grad():
            for k in ("mean3d", "log_scale", "quat", "opacity_raw", "sh"):
                getattr(gm, k).copy_(st[k].to(dev))
        gm.sh_degree_active = st["sh_degree_active"]
        print(f"\nusing trained cloud: {gm.n:,} gaussians (iter {ck['iter']})")
    else:
        gm = M.GaussianModel(100_000, 5.2, dev)
        print(f"\nusing random cloud: {gm.n:,} gaussians")

    print("\n" + "=" * 78)
    print("VARIANTS (forward + backward, 400x400, checkpointed)")
    print(f"{'variant':<22}{'s/iter':>9}{'speedup':>9}{'peak GB':>9}"
          f"{'img err':>11}{'grad rel':>11}{'warmup s':>10}")

    def show(r):
        print(f"{r['name']:<22}{r['s']:>9.3f}{base['s']/r['s']:>8.2f}x"
              f"{r['peak']:>9.2f}{r['err_img']:>11.2e}{r['err_grad']:>11.2e}"
              f"{r['warm']:>10.1f}", flush=True)

    base = run_variant("eager fp32", EAGER, gm, cam, chunk)
    ref = base["ref"]
    rows = [base]
    show(base)

    variants = [("bf16 (fp32 scan)", composite_bf16),
                ("bf16 (full)", composite_bf16_full)]

    # torch.compile: K differs for almost every chunk, so compiling on the raw
    # shapes triggers a recompilation storm that never amortises. Pad K to the
    # next power of two so only ~10 distinct shapes are ever seen. Padding is
    # safe: the extra slots get valid=0 -> alpha=0, which is the identity for
    # both the colour sum and the transmittance product.
    if args_compile:
        try:
            variants.insert(0, ("compile fp32 (pow2)",
                                pad_pow2(torch.compile(EAGER))))
            variants.append(("compile+bf16 (pow2)",
                             pad_pow2(torch.compile(composite_bf16))))
        except Exception as e:
            print(f"  (torch.compile unavailable: {str(e)[:50]})")
    # measure the cost of the padding itself, so the compile numbers are
    # attributable to fusion rather than to the bucketing
    variants.append(("eager fp32 (pow2 pad)", pad_pow2(EAGER)))

    for name, fn in variants:
        try:
            r = run_variant(name, fn, gm, cam, chunk, ref=ref)
            rows.append(r); show(r)
        except Exception as e:
            print(f"{name:<22}  FAILED: {str(e)[:60]}")
            torch.cuda.empty_cache()

    print("\nNOTE: 'img err' / 'grad err' are vs eager fp32. A variant is only")
    print("usable if the gradient error is small enough not to change training.")
    print("=" * 78)


if __name__ == "__main__":
    main()
