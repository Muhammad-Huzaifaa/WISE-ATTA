#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Running EATTA on ImageNet-C ==="
CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg cfgs/imagenet_c/eatta.yaml

echo "=== Running EATTA on ImageNet-A ==="
CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg cfgs/imagenet_a/eatta.yaml

echo "=== Running EATTA on ImageNet-K ==="
CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg cfgs/imagenet_k/eatta.yaml

echo "=== Running EATTA on ImageNet-R ==="
CUDA_VISIBLE_DEVICES=3 python test_time.py --cfg cfgs/imagenet_r/eatta.yaml

echo "=== All experiments completed successfully! ✅ ==="
