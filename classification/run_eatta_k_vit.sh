#!/bin/bash
set -e

GPU=3   # change this if you want another GPU

# Fixed step1 / base LR (your stable one)
LR1=0.001
BASE_LR=0.001
WD=0.0

# 10 values around 1e-5 for LR_STEP2
LRS2=(
  2.5e-06   # 0.0000025
  5e-06     # 0.000005
  7.5e-06   # 0.0000075
  1e-05     # 0.00001 (baseline)
  1.25e-05  # 0.0000125
  1.5e-05   # 0.000015
  2e-05     # 0.00002
  2.5e-05   # 0.000025
  3e-05     # 0.00003
  5e-05     # 0.00005
  1e-03     # 0.001
  1e-04     # 0.0001
  1e-02     # 0.01
)

mkdir -p logs

for LR2 in "${LRS2[@]}"; do
  echo "=== ImageNet-K - Vit | LR_STEP1=${LR1}, LR_STEP2=${LR2}, WD=${WD} ==="

  CUDA_VISIBLE_DEVICES=$GPU \
  python test_time.py \
    --cfg cfgs/imagenet_k/eatta_vit.yaml \
    OPTIM.LR $BASE_LR \
    OPTIM.LR_STEP1 $LR1 \
    OPTIM.LR_STEP2 $LR2 \
    OPTIM.WD $WD \

done

