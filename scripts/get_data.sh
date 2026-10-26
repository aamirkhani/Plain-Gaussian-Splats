#!/bin/bash
# Datasets are NOT redistributed in this repo - they carry their own licences.
set -e
mkdir -p data/lego/train data/lego/test notebook/data
B=https://huggingface.co/datasets/pablovela5620/nerf-synthetic-mirror/resolve/main/lego
echo "NeRF-synthetic 'lego' (CC-BY-NC, Heinzelnisse via BlendSwap)"
for f in transforms_train.json transforms_test.json; do curl -sL -o data/lego/$f $B/$f; done
seq 0 99 | xargs -P 8 -I{} curl -sL --retry 3 -o data/lego/train/r_{}.png $B/train/r_{}.png
seq 0 24 | xargs -P 8 -I{} curl -sL --retry 3 -o data/lego/test/r_{}.png  $B/test/r_{}.png
echo "Middlebury 2003 'cones' stereo pair (for the notebook)"
M=https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones
curl -sL -o notebook/data/left.png  $M/im2.png
curl -sL -o notebook/data/right.png $M/im6.png
echo "done: $(ls data/lego/train | wc -l) train views, $(ls data/lego/test | wc -l) test views"
