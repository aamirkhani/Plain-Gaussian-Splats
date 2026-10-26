"""Verify the Radeon 890M iGPU (gfx1150) under ROCm actually computes correctly.

We do NOT trust "device_count == 1". An unsupported/half-supported GPU can
return silently wrong numbers. So every test compares against a CPU reference.

The ops tested are exactly the ones the 3DGS rasterizer depends on:
  matmul            -> projection, covariance
  sort / argsort    -> per-tile depth ordering
  cumprod / cumsum  -> alpha compositing transmittance
  index_add / scatter-> gradient accumulation into per-gaussian buffers
  exp / sigmoid     -> opacity and gaussian falloff
  autograd backward -> the whole training loop

Run:  python src/00_verify_gpu.py
"""

import time

import torch

FAIL = []


def check(name, got, ref, tol, rel=False):
    """Compare a GPU result against its CPU reference.

    rel=True compares relative error - required for ops whose outputs span
    orders of magnitude (exp), where an absolute tolerance is meaningless.
    """
    got = got.detach().float().cpu()
    ref = ref.detach().float().cpu()
    diff = (got - ref).abs()
    if rel:
        err = (diff / ref.abs().clamp_min(1e-6)).max().item()
        label = "max_rel_err"
    else:
        err = diff.max().item()
        label = "max_abs_err"
    ok = err <= tol and torch.isfinite(got).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} {label}={err:.3e} (tol={tol:.1e})")
    if not ok:
        FAIL.append(name)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("ROCm torch reports no device - cannot use the AMD GPU.")

    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(0)
    print("=" * 70)
    print(f"device      : {p.name}")
    print(f"gcn arch    : {getattr(p, 'gcnArchName', '?')}")
    print(f"reported mem: {p.total_memory / 1024**3:.1f} GiB (shared with system RAM on an iGPU)")
    print(f"CUs         : {p.multi_processor_count}")
    print(f"torch       : {torch.__version__}  hip={torch.version.hip}")
    print("=" * 70)

    torch.manual_seed(0)

    # ---- 1. matmul ------------------------------------------------------
    print("\n[1] dense linear algebra")
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    check("matmul fp32", a.to(dev) @ b.to(dev), a @ b, 2e-3)
    # batched 3x3: exactly the shape of per-gaussian covariance products
    c3 = torch.randn(20_000, 3, 3)
    d3 = torch.randn(20_000, 3, 3)
    check("bmm fp32 (batched 3x3)",
          torch.bmm(c3.to(dev), d3.to(dev)), torch.bmm(c3, d3), 1e-4)

    # ---- 2. sort (tile depth ordering) ----------------------------------
    print("\n[2] sort / argsort  (per-tile depth ordering)")
    # Use a random permutation so all keys are DISTINCT. With duplicate keys the
    # permutation is not unique (GPU sort is not stable), so comparing indices
    # against CPU would flag a tie-break difference as a failure.
    k = torch.randperm(200_000).float()
    vg, ig = torch.sort(k.to(dev))
    vc, ic = torch.sort(k)
    check("sort values", vg, vc, 0.0)
    check("sort permutation (distinct keys)", ig.float(), ic.float(), 0.0)
    # Independent invariant: the returned permutation must actually sort the input.
    gathered = k.to(dev)[ig]
    check("key[argsort] == sorted(key)", gathered, vc, 0.0)
    # And with ties present, assert the weaker property that still must hold.
    kt = torch.randint(0, 50, (100_000,)).float()
    vt, it = torch.sort(kt.to(dev))
    check("tied keys: key[argsort] sorted", kt.to(dev)[it], torch.sort(kt)[0], 0.0)

    # ---- 3. cumprod / cumsum (alpha compositing) ------------------------
    print("\n[3] cumprod / cumsum  (transmittance along a ray)")
    al = torch.rand(4096, 64) * 0.5
    check("cumprod(1-alpha)", torch.cumprod(1 - al.to(dev), dim=1), torch.cumprod(1 - al, dim=1), 1e-5)
    check("cumsum", torch.cumsum(al.to(dev), dim=1), torch.cumsum(al, dim=1), 1e-3)

    # ---- 4. scatter / index_add (gradient accumulation) -----------------
    print("\n[4] index_add / scatter_add  (grad accumulation per gaussian)")
    n, m = 50_000, 300_000
    idx = torch.randint(0, n, (m,))
    src = torch.randn(m, 3)
    og = torch.zeros(n, 3, device=dev).index_add_(0, idx.to(dev), src.to(dev))
    oc = torch.zeros(n, 3).index_add_(0, idx, src)
    check("index_add_ (contended)", og, oc, 2e-4)

    # ---- 5. elementwise nonlinearities ---------------------------------
    print("\n[5] elementwise")
    x = torch.randn(100_000) * 3
    # exp spans many orders of magnitude -> relative tolerance, not absolute.
    check("exp", torch.exp(x.to(dev)), torch.exp(x), 1e-5, rel=True)
    # the gaussian falloff we actually evaluate is exp(-q) for q >= 0, in [0,1]
    q = torch.rand(100_000) * 12
    check("exp(-q) gaussian falloff", torch.exp(-q.to(dev)), torch.exp(-q), 1e-6)
    check("sigmoid", torch.sigmoid(x.to(dev)), torch.sigmoid(x), 1e-6)

    # ---- 6. autograd ----------------------------------------------------
    print("\n[6] autograd backward  (the actual training signal)")
    # Build the data ONCE on the CPU and copy it to each device. Seeding and
    # then calling randn(device=...) would use two different RNGs (CPU vs ROCm)
    # and silently compare two unrelated problems.
    w0 = torch.randn(256, 128)
    v0 = torch.randn(128, 64)
    inp0 = torch.randn(512, 256)
    tgt0 = torch.randn(512, 64)

    def run(device):
        w = w0.to(device).requires_grad_(True)
        v = v0.to(device).requires_grad_(True)
        inp, tgt = inp0.to(device), tgt0.to(device)
        loss = ((torch.relu(inp @ w) @ v - tgt) ** 2).mean()
        loss.backward()
        return loss, w.grad, v.grad

    lg, wg, vg2 = run(dev)
    lc, wc, vc2 = run("cpu")
    # Scale-relative tolerances: these tensors have magnitudes in the hundreds,
    # so fp32 reduction-order differences give absolute errors far above 1e-3.
    check("loss value", lg.view(1), lc.view(1), 1e-5, rel=True)
    check("grad w", wg, wc, 1e-4 * wc.abs().max().item())
    check("grad v", vg2, vc2, 1e-4 * vc2.abs().max().item())

    # ---- 7. an optimizer actually reduces loss --------------------------
    print("\n[7] 200 Adam steps on the GPU must reduce loss")
    torch.manual_seed(2)
    w = torch.randn(128, 64, device=dev, requires_grad=True)
    inp = torch.randn(256, 128, device=dev)
    tgt = torch.randn(256, 64, device=dev)
    opt = torch.optim.Adam([w], lr=1e-2)
    first = None
    for i in range(200):
        opt.zero_grad()
        loss = ((inp @ w - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    last = loss.item()
    ok = last < first * 0.5 and last == last
    print(f"  [{'PASS' if ok else 'FAIL'}] loss {first:.4f} -> {last:.4f}")
    if not ok:
        FAIL.append("adam optimization")

    # ---- 8. throughput --------------------------------------------------
    print("\n[8] throughput (sets expectations for training time)")
    for size in (1024, 2048):
        x1 = torch.randn(size, size, device=dev)
        for _ in range(3):
            x1 @ x1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            x1 @ x1
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        print(f"  {size}^3 matmul: {dt*1e3:7.2f} ms  -> {2*size**3/dt/1e12:.2f} TFLOP/s fp32")

    print(f"\n  peak GPU mem this run: {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")

    print("\n" + "=" * 70)
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
        print("=> Do NOT train on this GPU. Fall back to CPU.")
        raise SystemExit(1)
    print("RESULT: all checks passed - the 890M computes correctly. Safe to train.")
    print("=" * 70)


if __name__ == "__main__":
    main()
