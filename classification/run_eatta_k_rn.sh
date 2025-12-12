#!/bin/bash
set -e

GPU=3                    # choose GPU

# Ablation values around 0.95
MOMENTUMS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.80 0.90 0.93 0.95 0.97 0.99)

mkdir -p logs

for AM in "${MOMENTUMS[@]}"; do
  echo "=== ImageNet-K - RN | ANCHOR_MOMENTUM=${AM} ==="

  CUDA_VISIBLE_DEVICES=$GPU \
  python test_time.py \
    --cfg cfgs/imagenet_k/eatta_rn.yaml \
    MODEL.ANCHOR_MOMENTUM $AM \

done
