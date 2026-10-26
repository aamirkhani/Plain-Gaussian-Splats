"""Train 3D Gaussian Splatting on the AMD 890M iGPU under a hard memory cap.

Loss is the paper's   (1 - lambda) * L1  +  lambda * (1 - SSIM),  lambda = 0.2.
L1 alone gives blurry, washed-out results; the SSIM term is what recovers
structure.

Memory: the iGPU has no private VRAM, so every GPU allocation eats the same
pool as the CPU. --mem-gb caps the torch allocator so a runaway densification
cannot take the machine down.

Usage
  python src/train.py --bench                     # time a few iterations
  python src/train.py --iters 7000 --res 400      # real run
  python src/train.py --resume logs/ckpt_last.pt  # continue
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import model as M
import rasterizer as R

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# SSIM (11x11 gaussian window), on [H,W,3] inputs in [0,1]
# --------------------------------------------------------------------------
def _gauss_win(size=11, sigma=1.5, device="cpu"):
    c = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    return w.expand(3, 1, size, size).contiguous()


def ssim(a, b, win):
    """a, b: [H,W,3] -> scalar mean SSIM."""
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    pad = win.shape[-1] // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=3)
    mu_b = F.conv2d(b, win, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d(a * a, win, padding=pad, groups=3) - mu_a2
    sb = F.conv2d(b * b, win, padding=pad, groups=3) - mu_b2
    sab = F.conv2d(a * b, win, padding=pad, groups=3) - mu_ab
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_ab + C1) * (2 * sab + C2)) / \
        ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return m.mean()


# --------------------------------------------------------------------------
def render_cam(gm, cam, chunk_elems, use_ckpt=True):
    return R.render(gm.mean3d, gm.log_scale, gm.quat, gm.opacity_raw, gm.sh,
                    cam.viewmat, cam.cam_pos, cam.H, cam.W,
                    cam.fx, cam.fy, cam.cx, cam.cy,
                    sh_degree=gm.sh_degree_active, bg=1.0,
                    chunk_elems=chunk_elems, use_checkpoint=use_ckpt)


@torch.no_grad()
def evaluate(gm, cams, chunk_elems, out_dir=None, tag="", max_views=None):
    vals = []
    views = cams if max_views is None else cams[:max_views]
    for i, c in enumerate(views):
        img, _, _ = render_cam(gm, c, chunk_elems, use_ckpt=False)
        img = img.clamp(0, 1)
        vals.append(D.psnr(img, c.image))
        if out_dir is not None and i < 3:
            _save_png(img, os.path.join(out_dir, f"{tag}_view{i}.png"))
    return sum(vals) / max(len(vals), 1), vals


def _save_png(img, path):
    from PIL import Image
    a = (img.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(a).save(path)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "data/lego"))
    p.add_argument("--out", default=os.path.join(HERE, "logs"))
    p.add_argument("--res", type=int, default=400)
    p.add_argument("--iters", type=int, default=7000)
    p.add_argument("--init-points", type=int, default=30_000)
    p.add_argument("--cap", type=int, default=150_000, help="max gaussians")
    p.add_argument("--mem-gb", type=float, default=8.0,
                   help="hard cap on the torch GPU allocator")
    p.add_argument("--chunk-elems", type=int, default=6_000_000)
    p.add_argument("--densify-from", type=int, default=500)
    p.add_argument("--densify-until-frac", type=float, default=0.5)
    p.add_argument("--densify-every", type=int, default=100)
    p.add_argument("--grad-thresh", type=float, default=2e-4)
    p.add_argument("--opacity-reset-every", type=int, default=3000)
    p.add_argument("--sh-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-cloud", default=None,
                   help="path to a .pt with xyz/rgb to initialise from")
    p.add_argument("--init-subsample", type=int, default=0,
                   help="randomly keep only N points of the init cloud")
    p.add_argument("--init-opacity", type=float, default=0.1)
    p.add_argument("--init-normals", action="store_true",
                   help="surface-align gaussians using normals in the cloud")
    p.add_argument("--depth-guide", default=None,
                   help="path to depthmaps.pt -> enable depth-guided density control")
    p.add_argument("--dg-prune", action="store_true", help="prune floaters")
    p.add_argument("--dg-infill", action="store_true", help="seed uncovered surface")
    p.add_argument("--dg-thresh", type=float, default=0.6)
    p.add_argument("--flat", type=float, default=0.25,
                   help="how flat a surface-aligned gaussian starts")
    p.add_argument("--fuse", action="store_true",
                   help="Triton-fuse the pointwise prefix (~1.86x on the 890M)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.fuse:
        print("kernel fusion:", "enabled" if R.enable_fusion() else "FAILED")
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                       else "cuda:0")

    if dev.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        frac = min(args.mem_gb / total, 1.0)
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"device {torch.cuda.get_device_properties(0).name} "
              f"| allocator capped at {args.mem_gb:.1f} GiB "
              f"({frac*100:.0f}% of {total:.1f} GiB shared)")

    # ---- data --------------------------------------------------------
    t0 = time.perf_counter()
    train_cams = D.load_blender(args.data, "train", args.res, dev)
    test_cams = D.load_blender(args.data, "test", args.res, dev, limit=8)
    extent = D.scene_extent(train_cams)
    print(f"loaded {len(train_cams)} train / {len(test_cams)} test views "
          f"at {args.res}x{args.res} in {time.perf_counter()-t0:.1f}s "
          f"| scene extent {extent:.2f}")

    if args.init_cloud:
        cl = torch.load(args.init_cloud, map_location="cpu", weights_only=False)
        xyz, rgb = cl["xyz"], cl["rgb"]
        if args.init_subsample and args.init_subsample < xyz.shape[0]:
            g = torch.Generator().manual_seed(args.seed)
            keep = torch.randperm(xyz.shape[0], generator=g)[:args.init_subsample]
            xyz, rgb = xyz[keep], rgb[keep]
        nrm = cl.get("normal")
        if nrm is not None and args.init_subsample and args.init_subsample < cl["xyz"].shape[0]:
            nrm = nrm[keep]
        gm = M.GaussianModel(0, extent, dev, init_xyz=xyz, init_rgb=rgb,
                             init_opacity=args.init_opacity,
                             init_normal=nrm if args.init_normals else None,
                             flat=args.flat)
        print(f"initialised from {args.init_cloud}: {gm.n:,} gaussians "
              f"(opacity {args.init_opacity})")
    else:
        gm = M.GaussianModel(args.init_points, extent, dev,
                             init_opacity=args.init_opacity)
        print(f"random initialisation: {gm.n:,} gaussians "
              f"(opacity {args.init_opacity})")
    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        gm = M.GaussianModel(ck["state"]["mean3d"].shape[0], ck["state"]["extent"], dev)
        with torch.no_grad():
            for k in ("mean3d", "log_scale", "quat", "opacity_raw", "sh"):
                getattr(gm, k).copy_(ck["state"][k].to(dev))
        gm.sh_degree_active = ck["state"]["sh_degree_active"]
        start_it = ck["iter"]
        print(f"resumed from {args.resume} at iter {start_it} "
              f"with {gm.n} gaussians")

    DG = None
    if args.depth_guide:
        import depth_guide as DGm
        _d = torch.load(args.depth_guide, map_location=dev, weights_only=False)
        DG = (DGm, _d["depth"].float().to(dev), _d["mask"].to(dev))
        print(f"depth-guided density control: prune={args.dg_prune} "
              f"infill={args.dg_infill}")

    win = _gauss_win(device=dev)
    densify_until = int(args.iters * args.densify_until_frac)
    log_path = os.path.join(args.out, "train_log.jsonl")
    logf = open(log_path, "a")

    # ---- benchmark mode ---------------------------------------------
    if args.bench:
        print("\nbenchmark (forward+backward):")
        for npts in (30_000, 100_000, 200_000):
            g = M.GaussianModel(npts, extent, dev)
            cam = train_cams[0]
            for _ in range(2):   # warmup
                img, _, st = render_cam(g, cam, args.chunk_elems)
                ((img - cam.image).abs().mean()).backward()
                g.opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            reps = 5
            for _ in range(reps):
                img, _, st = render_cam(g, cam, args.chunk_elems)
                loss = 0.8 * (img - cam.image).abs().mean() + \
                    0.2 * (1 - ssim(img, cam.image, win))
                loss.backward()
                g.opt.step()
                g.opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t) / reps
            peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
                    if dev.type == "cuda" else 0.0)
            print(f"  {npts:>7,} gaussians: {dt:6.3f} s/iter  "
                  f"peak {peak:.2f} GiB  pairs={st.get('n_pairs',0):,}  "
                  f"-> {args.iters} iters = {dt*args.iters/3600:.2f} h")
            del g
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        return

    # ---- training loop ----------------------------------------------
    print(f"\ntraining {args.iters} iters | densify {args.densify_from}.."
          f"{densify_until} every {args.densify_every} | cap {args.cap:,}")
    perm = torch.randperm(len(train_cams)).tolist()
    pi = 0
    ema = None
    t_start = time.perf_counter()

    for it in range(start_it + 1, args.iters + 1):
        gm.update_lr(it, args.iters)
        if it % args.sh_every == 0:
            gm.oneup_sh_degree()

        if pi >= len(perm):
            perm = torch.randperm(len(train_cams)).tolist()
            pi = 0
        cam = train_cams[perm[pi]]
        pi += 1

        img, acc, stats = render_cam(gm, cam, args.chunk_elems)
        l1 = (img - cam.image).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim(img, cam.image, win))
        loss.backward()

        with torch.no_grad():
            gm.record_stats(stats, cam.W)
        gm.opt.step()
        gm.opt.zero_grad(set_to_none=True)

        lv = float(loss.item())
        ema = lv if ema is None else 0.95 * ema + 0.05 * lv
        if not math.isfinite(lv):
            raise RuntimeError(f"loss became {lv} at iter {it} - aborting")

        # ---- adaptive density control -------------------------------
        if args.densify_from <= it <= densify_until and it % args.densify_every == 0:
            info = gm.densify_and_prune(
                grad_thresh=args.grad_thresh, min_opacity=0.005,
                max_screen=None, cap=args.cap)
            msg = (f"  it {it:6d} densify: {info['before']:,} "
                   f"+{info['cloned']:,}c +{info['split']:,}s "
                   f"-{info['pruned']:,}p -> {info['after']:,}")
            if DG is not None:
                DGm, dmap, dmask = DG
                gi = DGm.apply_depth_guidance(
                    gm, train_cams, dmap, dmask,
                    lambda g, c: render_cam(g, c, args.chunk_elems, use_ckpt=False),
                    do_prune=args.dg_prune, do_infill=args.dg_infill,
                    float_thresh=args.dg_thresh, cap=args.cap)
                msg += f"  | depth-guide -{gi['floaters']:,}f +{gi['infilled']:,}i -> {gm.n:,}"
            print(msg)
        if (args.opacity_reset_every and it % args.opacity_reset_every == 0
                and it <= densify_until):
            gm.reset_opacity()
            print(f"  it {it:6d} opacity reset")

        # ---- logging ------------------------------------------------
        if it % 100 == 0 or it == 1:
            el = time.perf_counter() - t_start
            done = it - start_it
            peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
                    if dev.type == "cuda" else 0.0)
            msg = {"iter": it, "loss": round(lv, 5), "ema": round(ema, 5),
                   "l1": round(float(l1.item()), 5), "n": gm.n,
                   "sh": gm.sh_degree_active,
                   "s_per_it": round(el / max(done, 1), 3),
                   "peak_gb": round(peak, 2),
                   "eta_h": round((args.iters - it) * el / max(done, 1) / 3600, 2)}
            print(f"it {it:6d} loss {lv:.4f} ema {ema:.4f} n {gm.n:,} "
                  f"{el/max(done,1):.2f}s/it peak {peak:.2f}GB "
                  f"eta {msg['eta_h']:.2f}h", flush=True)
            logf.write(json.dumps(msg) + "\n")
            logf.flush()

        # ---- eval / checkpoint --------------------------------------
        if it % args.eval_every == 0 or it == args.iters:
            ps, _ = evaluate(gm, test_cams, args.chunk_elems,
                             out_dir=args.out, tag=f"it{it:06d}", max_views=8)
            tr, _ = evaluate(gm, train_cams[:8], args.chunk_elems, max_views=8)
            print(f"  >> it {it}: test PSNR {ps:.2f} dB | train PSNR {tr:.2f} dB",
                  flush=True)
            logf.write(json.dumps({"iter": it, "test_psnr": round(ps, 3),
                                   "train_psnr": round(tr, 3), "n": gm.n}) + "\n")
            logf.flush()
        if it % args.save_every == 0 or it == args.iters:
            torch.save({"iter": it, "state": gm.state(), "args": vars(args)},
                       os.path.join(args.out, "ckpt_last.pt"))

    ps, per = evaluate(gm, test_cams, args.chunk_elems, out_dir=args.out,
                       tag="final")
    print(f"\nFINAL test PSNR {ps:.2f} dB over {len(per)} views | "
          f"{gm.n:,} gaussians | {(time.perf_counter()-t_start)/3600:.2f} h")
    logf.close()


if __name__ == "__main__":
    main()
