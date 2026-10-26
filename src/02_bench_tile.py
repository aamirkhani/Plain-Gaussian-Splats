"""Find the best tile size and resolution for this iGPU.

Per-gaussian compositing work is  P * tiles_touched  =  (2r + TILE)^2,
so a smaller tile fits the 3-sigma ellipse more tightly and does strictly
less arithmetic - at the cost of more tiles, more binning pairs and more
kernel launches. Where that trade lands is hardware specific, so measure.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import model as M
import rasterizer as R
from train import _gauss_win, render_cam, ssim

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bench(cam, gm, win, chunk, reps=4):
    for _ in range(2):
        img, _, st = render_cam(gm, cam, chunk)
        (0.8 * (img - cam.image).abs().mean()
         + 0.2 * (1 - ssim(img, cam.image, win))).backward()
        gm.opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(reps):
        img, _, st = render_cam(gm, cam, chunk)
        (0.8 * (img - cam.image).abs().mean()
         + 0.2 * (1 - ssim(img, cam.image, win))).backward()
        gm.opt.step()
        gm.opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / reps
    return dt, st.get("n_pairs", 0), torch.cuda.max_memory_allocated() / 1024 ** 3


def main():
    dev = torch.device("cuda:0")
    npts = 100_000
    print(f"{'res':>5} {'tile':>5} {'s/iter':>8} {'pairs':>10} {'peak GB':>8} "
          f"{'7k iters (h)':>13}")
    for res in (200, 400):
        cams = D.load_blender(os.path.join(HERE, "data/lego"), "train", res,
                              dev, limit=1)
        extent = 5.2
        win = _gauss_win(device=dev)
        for tile in (4, 8, 16):
            R.TILE = tile
            gm = M.GaussianModel(npts, extent, dev)
            try:
                dt, pairs, peak = bench(cams[0], gm, win, 16_000_000)
                print(f"{res:>5} {tile:>5} {dt:>8.3f} {pairs:>10,} {peak:>8.2f} "
                      f"{dt*7000/3600:>13.2f}")
            except RuntimeError as e:
                print(f"{res:>5} {tile:>5}   FAILED: {str(e)[:60]}")
            del gm
            torch.cuda.empty_cache()
        del cams
        torch.cuda.empty_cache()
    R.TILE = 16


if __name__ == "__main__":
    main()
