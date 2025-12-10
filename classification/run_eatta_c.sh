#!/bin/bash
set -e

GPU=3   # change this if you want another GPU

# Grid for LR_STEP1 (around your current 2.5e-4)
LRS2=(
  1e-6
  1e-5
  1e-4
  1e-3
)

# We'll keep global LR equal to LR_STEP1 (like before)
LR1=6e-4   
WD=0.0

mkdir -p logs

for LRS2 in "${LRS2[@]}"; do
  echo "=== ImageNet-C | LR_STEP1=${LR1}, LR_STEP2=${LRS2}, WD=${WD} ==="

  CUDA_VISIBLE_DEVICES=$GPU \
  python test_time.py \
    --cfg cfgs/imagenet_c/eatta.yaml \
    OPTIM.LR $LR1 \
    OPTIM.LR_STEP1 $LR1 \
    OPTIM.LR_STEP2 $LRS2 \
    OPTIM.WD $WD


done
