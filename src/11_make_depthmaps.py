"""Save per-view metric depth maps for depth-guided density control.

Deliberately a SEPARATE script from 10_depth_init.py. Re-running that would
re-solve the object radius R, and R is not reproducible across runs (we have
observed 0.50 / 0.60 / 0.70) because the photo-consistency curve is nearly
flat and its sampler draws from the CUDA RNG. A fresh solve would therefore
rewrite cloud_surface.pt with different geometry and silently break
comparability with every round-3 and round-4 result.

So: reuse the EXACT R that cloud_surface.pt was built with, and touch nothing
else.
"""

import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import importlib.util

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "di", os.path.join(os.path.dirname(os.path.abspath(__file__)), "10_depth_init.py"))
di = importlib.util.module_from_spec(spec)
spec.loader.exec_module(di)


def main():
    dev = torch.device("cuda")
    out = os.path.join(HERE, "logs/depth_init")
    cloud = torch.load(os.path.join(out, "cloud_surface.pt"),
                       map_location="cpu", weights_only=False)
    R = float(cloud["R"])
    print(f"reusing R = {R:.2f} from cloud_surface.pt (NOT re-solving)")

    cams = D.load_blender(os.path.join(HERE, "data/lego"), "train", 400, dev)
    meta = json.load(open(os.path.join(HERE, "data/lego/transforms_train.json")))
    paths = [os.path.join(HERE, "data/lego", f["file_path"].lstrip("./") + ".png")
             for f in meta["frames"]]

    pred, mask = di.predict_depth(cams, paths, dev)
    pnorm = di.normalise_pred(pred, mask)
    dmaps = torch.stack([di.depth_from_radius(pnorm[v], cams[v].cam_pos.norm(), R)
                         for v in range(len(cams))])

    torch.save({"depth": dmaps.half().cpu(), "mask": mask.cpu(), "R": R},
               os.path.join(out, "depthmaps.pt"))
    print(f"saved depthmaps.pt {tuple(dmaps.shape)}  "
          f"depth {float(dmaps.min()):.2f}..{float(dmaps.max()):.2f}")

    # sanity: the cloud's own points should sit AT the predicted surface, not
    # in front of it - if they do not, the maps and the cloud disagree.
    xyz = cloud["xyz"].to(dev)
    c = cams[0]
    Rm, t = c.viewmat[:3, :3], c.viewmat[:3, 3]
    pc = xyz @ Rm.T + t
    z = pc[:, 2]
    u = c.fx * pc[:, 0] / z.clamp(min=1e-4) + c.cx
    v = c.fy * pc[:, 1] / z.clamp(min=1e-4) + c.cy
    H, W = c.H, c.W
    inb = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    fl = (v.long().clamp(0, H - 1) * W + u.long().clamp(0, W - 1))
    on = mask[0].reshape(-1)[fl] & inb
    diff = (z - dmaps[0].reshape(-1)[fl])[on]
    print(f"cloud-vs-depthmap agreement on view 0: "
          f"median {float(diff.median()):+.3f}  "
          f"|diff|<0.05 for {float((diff.abs() < 0.05).float().mean())*100:.0f}% of points")


if __name__ == "__main__":
    main()
