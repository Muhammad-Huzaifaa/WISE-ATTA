#!/bin/bash

echo "Running for continual test-time adaptation on ImageNet-C with EATTA"

SEEDS=(0 41 58)
CONFIG=cfgs/imagenet_c/eatta_ctta.yaml

for SEED in "${SEEDS[@]}"; do
    echo "Running seed = $SEED"
    CUDA_VISIBLE_DEVICES=3 python test_time.py \
        --cfg "$CONFIG" \
        SEED "$SEED"
done

