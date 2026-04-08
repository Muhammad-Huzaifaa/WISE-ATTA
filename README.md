# WISE-ATTA

Official PyTorch implementation of  
**WISE-ATTA: When to Ask for Labels in Budgeted Active Test-Time Adaptation**

> Budget-aware active test-time adaptation for long test streams with limited supervision.

![Framework](frameworks.png)

---

## Overview

**WISE-ATTA** studies a practical setting of **budgeted active test-time adaptation (ATTA)**, where labels are available for only a small fraction of test batches rather than every batch. The key question is not only **what to label**, but also **when to spend the labeling budget** over time. :contentReference[oaicite:0]{index=0}

To address this, WISE-ATTA introduces two complementary components:  
- **Budget-paced batch selection**, which decides **when** to request supervision using lightweight online utility signals.  
- **Drift-based sample selection**, which decides **what** to label by selecting a single informative sample based on prediction drift relative to an EMA anchor model. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

Across **ImageNet-C** and natural distribution shift benchmarks (**ImageNet-R**, **ImageNet-K**, and **ImageNet-A**), WISE-ATTA achieves competitive or improved robustness while using substantially fewer labels. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

## Highlights

- **Budgeted ATTA** for long test streams
- **Selective supervision over time** instead of labeling every batch
- **Single-sample querying** for selected batches
- No replay buffer required
- Evaluated on **ImageNet-C/R/K/A** :contentReference[oaicite:5]{index=5}

---

## Environment

Tested with:

- **Driver Version:** 550.67  
- **CUDA Version:** 12.4  
- **Python Version:** 3.10.12  

We provide a conda environment:

```bash
conda update conda
conda env create -f environment.yml
conda activate wise-atta


## Datasets

We evaluate on the following benchmarks:

- **ImageNet-C**
- **ImageNet-R**
- **ImageNet-K / ImageNet-Sketch**
- **ImageNet-A**

After downloading the datasets, update the dataset root in `conf.py`:

```python
_C.DATA_DIR = "/your/dataset/path"


Method

WISE-ATTA operates in two stages:

1. Batch Selection

For each incoming test batch, WISE-ATTA estimates whether supervision is likely to be useful and decides whether to spend part of the label budget on that batch.

2. Sample Selection

If a batch is selected, WISE-ATTA queries the label of one sample whose prediction shows large drift relative to an EMA anchor model, indicating ongoing but unconverged adaptation.

This design makes label usage more efficient under constrained annotation budgets.


Run on ImageNet-C
cd classification
python test_time.py --cfg cfgs/imagenet_c/wiseatta.yaml MODEL.ADAPTATION wise
Batch Selection Strategies

You can replace wise in MODEL.ADAPTATION with:

wise — WISE-ATTA batch selection
uniform — uniform batch selection
random — random batch selection

Example:

python test_time.py --cfg cfgs/imagenet_c/wiseatta.yaml MODEL.ADAPTATION uniform
Configuration

You may want to modify the following options in ./cfgs/imagenet_c/wiseatta.yaml:

MODEL:
  ADAPTATION: wiseatta
  EDGE_CLOUD: True
  CLOUD_ARCH: 'vit_l_16'          # Annotator / large model architecture if used
  EDGE_ARCH: 'resnet50'           # Adapted model architecture
  EDGE_ARCH_WEIGHTS: 'IMAGENET1K_V1'
  HUMAN_OR_LARGE_MODEL: 'HUMAN'   # 'HUMAN' or 'LARGE_MODELS'
  ORACLE_NUM: 1                   # Number of queried labels for a selected batch
  BUFFER: False
Acknowledgements

This repository is built upon the excellent codebases:

EATTA
test-time-adaptation

We also thank the authors of prior TTA and ATTA methods that inspired this work, including TENT, SimATTA, CEMA, HILTTA, and EATTA.

Contact

For questions or collaborations, please contact:

Muhammad Huzaifa
muhammad.huzaifa [at] cispa.de