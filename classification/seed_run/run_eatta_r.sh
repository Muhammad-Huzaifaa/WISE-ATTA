#!/bin/bash

SEEDS=(0 41 58)
CONFIG=cfgs/imagenet_r/eatta_rn.yaml

echo "Running for fully test-time adaptation on ImageNet-R on resnet with EATTA"

# for SEED in "${SEEDS[@]}"; do
#     echo "Running seed = $SEED"
#     CUDA_VISIBLE_DEVICES=3 python test_time.py \
#         --cfg "$CONFIG" \
#         SEED "$SEED"
# done


CONFIG=cfgs/imagenet_r/eatta_vit.yaml
echo "Running for fully test-time adaptation on ImageNet-R on vit with EATTA"

for SEED in "${SEEDS[@]}"; do
    echo "Running seed = $SEED"
    CUDA_VISIBLE_DEVICES=3 python test_time.py \
        --cfg "$CONFIG" \
        SEED "$SEED"
done

