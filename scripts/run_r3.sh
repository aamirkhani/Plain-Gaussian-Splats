#!/bin/bash
# Round 3: does SURFACE-ALIGNED initialisation beat isotropic?
# Identical points in every arm (cloud_surface, 25k subsample, cap 40k).
# The ONLY difference is whether each gaussian starts as a sphere or as a
# disc lying in the local tangent plane.
PY=/home/akhani/miniconda3/envs/gsplat_rocm/bin/python   # AMD 890M only
C="--res 400 --iters 2000 --cap 40000 --densify-from 200 --densify-until-frac 0.6 \
--densify-every 100 --eval-every 250 --save-every 2000 --seed 7 --fuse --chunk-elems 16000000 \
--init-cloud logs/depth_init/cloud_surface.pt --init-subsample 25000"
run () { echo "=============== ARM: $1 ==============="; \
  $PY -u src/train.py $C --out logs/r3_$1 "${@:2}" 2>&1 \
  | stdbuf -oL grep -E "initialis|>>|FINAL"; }
run iso                                      # control: isotropic spheres
run flat25 --init-normals --flat 0.25        # disc: 4x flatter along normal
run flat10 --init-normals --flat 0.10        # very flat
run flat50 --init-normals --flat 0.50        # mildly flattened
echo "ROUND 3 DONE"
