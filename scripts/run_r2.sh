#!/bin/bash
# Round 2: remove the capacity confound, then probe two refinements.
# EVERY arm: 25k starting points, hard cap 40k, same seed, same schedule.
PY=/home/akhani/miniconda3/envs/gsplat_rocm/bin/python   # AMD 890M only
C="--res 400 --iters 2000 --cap 40000 --densify-from 200 --densify-until-frac 0.6 \
--densify-every 100 --eval-every 250 --save-every 2000 --seed 7 --fuse --chunk-elems 16000000"
DEPTH="--init-cloud logs/depth_init/cloud_voxel.pt --init-subsample 25000"
run () { echo "=============== ARM: $1 ==============="; \
  $PY -u src/train.py $C --out logs/r2_$1 "${@:2}" 2>&1 \
  | stdbuf -oL grep -E "initialis|random init|>>|FINAL"; }

# control: identical capacity, only the init differs
run m_random  --init-points 25000
run m_depth   $DEPTH
# refinement 1: depth points sit on surfaces, so let them contribute sooner
run m_opacity $DEPTH --init-opacity 0.5
# refinement 2: fewer but well-placed seeds, let densification do more
run m_lean    --init-cloud logs/depth_init/cloud_voxel.pt --init-subsample 10000
echo "ROUND 2 DONE"
