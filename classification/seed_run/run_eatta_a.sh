#!/bin/bash

SEEDS=(0 41 58)
CONFIG=cfgs/imagenet_a/eatta_rn.yaml


echo "Running for fully test-time adaptation on ImageNet-A on resnet with EATTA"

for SEED in "${SEEDS[@]}"; do
    echo "Running seed = $SEED"
    CUDA_VISIBLE_DEVICES=3 python test_time.py \
        --cfg "$CONFIG" \
        SEED "$SEED"
done

CONFIG=cfgs/imagenet_a/eatta_vit.yaml

echo "Running for fully test-time adaptation on ImageNet-A on vit with EATTA"

for SEED in "${SEEDS[@]}"; do
    echo "Running seed = $SEED"
    CUDA_VISIBLE_DEVICES=0 python test_time.py \
        --cfg "$CONFIG" \
        SEED "$SEED"
done