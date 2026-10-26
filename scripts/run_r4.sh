#!/bin/bash
# Round 4: the ceiling is NOT the initialisation - is it the densification?
# Every arm starts from the SAME best-known init (surface-aligned, flat 0.10,
# 25k seeds). Only the density-control policy changes.
# Reference point: that init at cap 40k scored 25.82 dB.
PY=/home/akhani/miniconda3/envs/gsplat_rocm/bin/python   # AMD 890M only
BASE="--res 400 --iters 2000 --densify-from 200 --densify-every 100 \
--eval-every 250 --save-every 2000 --seed 7 --fuse --chunk-elems 16000000 \
--init-cloud logs/depth_init/cloud_surface.pt --init-subsample 25000 \
--init-normals --flat 0.10"
run () { echo "=============== ARM: $1 ==============="; \
  $PY -u src/train.py $BASE --out logs/r4_$1 "${@:2}" 2>&1 \
  | stdbuf -oL grep -E "initialis|>>|FINAL"; }

# 1. is the ceiling simply the gaussian budget?
run cap80     --cap 80000  --densify-until-frac 0.6
# 2. densify harder into the same budget
run grad1e4   --cap 40000  --densify-until-frac 0.6 --grad-thresh 1e-4
# 3. keep growing for longer
run window85  --cap 40000  --densify-until-frac 0.85
# 4. mid-run opacity reset to clear floaters and re-grow
run reset1000 --cap 40000  --densify-until-frac 0.6 --opacity-reset-every 1000
echo "ROUND 4 DONE"
