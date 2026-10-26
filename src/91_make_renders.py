"""Render the figures that show what the model actually produces."""
import os, sys
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D, model as M, rasterizer as R

OUT = "docs/figures"; dev = torch.device("cuda")
ck = torch.load("logs/run400/ckpt_last.pt", map_location="cpu", weights_only=False)
st = ck["state"]
gm = M.GaussianModel(st["mean3d"].shape[0], st["extent"], dev)
with torch.no_grad():
    for k in ("mean3d","log_scale","quat","opacity_raw","sh"): getattr(gm,k).copy_(st[k].to(dev))
gm.sh_degree_active = st["sh_degree_active"]
cams = D.load_blender("data/lego", "test", 400, dev, limit=3)

def rend(g, c):
    with torch.no_grad():
        img,_,_ = R.render(g.mean3d,g.log_scale,g.quat,g.opacity_raw,g.sh,c.viewmat,
                           c.cam_pos,c.H,c.W,c.fx,c.fy,c.cx,c.cy,
                           sh_degree=g.sh_degree_active,bg=1.0,use_checkpoint=False)
    return img.clamp(0,1)

def strip(tiles, labels, path, scale=1):
    H,W,_ = tiles[0].shape; pad, lab = 10, 26
    cv = Image.new("RGB",(W*len(tiles)+pad*(len(tiles)-1), H+lab),"white")
    d = ImageDraw.Draw(cv)
    for i,(t,l) in enumerate(zip(tiles,labels)):
        cv.paste(Image.fromarray((t*255).astype(np.uint8)), (i*(W+pad),0))
        d.text((i*(W+pad)+4, H+6), l, fill=(16,21,27))
    if scale!=1: cv = cv.resize((cv.width*scale, cv.height*scale), Image.LANCZOS)
    cv.save(path); print("wrote", path)

# --- hero: reconstruction vs truth -----------------------------------------
c = cams[0]; img = rend(gm,c)
err = ((img-c.image).abs()*4).clamp(0,1)
psnr = D.psnr(img, c.image)
strip([img.cpu().numpy(), c.image.cpu().numpy(), err.cpu().numpy()],
      [f"reconstruction  ({psnr:.2f} dB)", "ground truth", "error x4"],
      f"{OUT}/fig5_reconstruction.png")

# --- how the scene is built out of gaussians -------------------------------
op = torch.sigmoid(gm.opacity_raw).reshape(-1)
s = torch.exp(gm.log_scale); ssort = s.sort(dim=1, descending=True).values
imp = (op*ssort[:,0]*ssort[:,1]).argsort(descending=True)
tiles, labels = [], []
for k in (500, 5000, 40000, gm.n):
    sub = M.GaussianModel(0, gm.extent, dev,
                          init_xyz=gm.mean3d.detach()[imp[:k]],
                          init_rgb=torch.zeros(k,3,device=dev))
    with torch.no_grad():
        for a in ("log_scale","quat","opacity_raw","sh"):
            getattr(sub,a).copy_(getattr(gm,a).detach()[imp[:k]])
    sub.sh_degree_active = gm.sh_degree_active
    tiles.append(rend(sub,c).cpu().numpy()); labels.append(f"{k:,} gaussians")
strip(tiles, labels, f"{OUT}/fig6_buildup.png")
