#!/bin/bash
PY=/home/akhani/miniconda3/envs/gsplat_rocm/bin/python   # AMD 890M only
COMMON="--res 400 --iters 2000 --cap 150000 --densify-from 200 --densify-until-frac 0.6 \
--densify-every 100 --eval-every 250 --save-every 2000 --seed 7 --fuse --chunk-elems 16000000"
run () {
  echo "=============== ARM: $1 ==============="
  $PY -u src/train.py $COMMON --out logs/ab_$1 "${@:2}" 2>&1 \
    | stdbuf -oL grep -E "initialised|random init|>>|FINAL"
}
run random   --init-points 50000
run uniform  --init-cloud logs/depth_init/cloud_uniform.pt
run voxel    --init-cloud logs/depth_init/cloud_voxel.pt
run hybrid   --init-cloud logs/depth_init/cloud_hybrid.pt
echo "ALL ARMS DONE"
