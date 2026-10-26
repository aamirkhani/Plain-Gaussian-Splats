#!/bin/bash
# Round 5: the colour ceiling vs depth-guided density control.
# Rounds 1-4 showed EVERY geometry intervention converges to 25.7-25.9 dB.
# Diagnosis: with sh-every=1000 over 2000 iters the model sits at SH degree 0
# for half of training and never reaches degree 3 - a COLOUR ceiling, not a
# geometry one. Arms 1-2 test that. Arms 3-4 test depth-guided density control.
# All start from the best-known init (surface-aligned flat 0.10, 25k seeds).
# Reference: that config scored 25.82 dB.
PY=/home/akhani/miniconda3/envs/gsplat_rocm/bin/python   # AMD 890M only
B="--res 400 --iters 2000 --cap 40000 --densify-from 200 --densify-until-frac 0.6 \
--densify-every 100 --eval-every 250 --save-every 2000 --seed 7 --fuse --chunk-elems 16000000 \
--init-cloud logs/depth_init/cloud_surface.pt --init-subsample 25000 --init-normals --flat 0.10"
run () { echo "=============== ARM: $1 ==============="; \
  $PY -u src/train.py $B --out logs/r5_$1 "${@:2}" 2>&1 \
  | stdbuf -oL grep -E "initialis|depth-guided|>>|FINAL"; }

run sh300  --sh-every 300            # SH degree 3 by iter 900
run sh150  --sh-every 150            # SH degree 3 by iter 450
# floater-PRUNING dropped: validated before running and it cannot discriminate
# (depth-derived points scored 0.424, random points 0.406 - no signal). The
# depth prior's own error (median 0.105) exceeds the tolerance it would need.
# INFILL does not depend on that detector - it compares rendered alpha to the
# object mask - so it is still worth testing.
run dginfill --depth-guide logs/depth_init/depthmaps.pt --dg-infill
run sh300_infill --sh-every 300 --depth-guide logs/depth_init/depthmaps.pt --dg-infill
echo "ROUND 5 DONE"
