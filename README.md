# WISE-ATTA

Official PyTorch implementation of  
**WISE-ATTA: When to Ask for Labels in Budgeted Active Test-Time Adaptation**

> Budget-aware active test-time adaptation for long test streams with limited supervision.

<p align="center">
  <img src="frameworks.png" alt="Framework" width="500"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Environment](#environment)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Batch Selection Strategies](#batch-selection-strategies)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
- [Citation](#citation)

---

## Overview

**WISE-ATTA** studies a practical setting of **budgeted active test-time adaptation (ATTA)**, where labels are available for only a small fraction of test batches rather than every batch. The key question is not only **what to label**, but also **when to spend the labeling budget** over time.

To address this, WISE-ATTA introduces two complementary components:
- **Budget-paced batch selection**, which decides **when** to request supervision using lightweight online utility signals.
- **Drift-based sample selection**, which decides **what** to label by selecting a single informative sample based on prediction drift relative to an EMA anchor model.

---

## Highlights

- **Budgeted ATTA** for long test streams
- **Selective supervision over time** instead of labeling every batch
- **Single-sample querying** for selected batches
- No replay buffer required
- Evaluated on **ImageNet-C/R/K/A**

---

## Environment

Tested with:

- **Driver Version:** 550.67
- **CUDA Version:** 12.4
- **Python Version:** 3.10.12

---

## Datasets

We evaluate on the following benchmarks:

- **[ImageNet-C](https://zenodo.org/records/2235448#.Yj2RO_co_mF)**: Common corruptions
- **[ImageNet-R](https://github.com/hendrycks/imagenet-r)**: Renditions
- **[ImageNet-K / ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)**: Sketches
- **[ImageNet-A](https://github.com/hendrycks/natural-adv-examples)**: Adversarial examples

After downloading the datasets, update the dataset root in `classification/conf.py`:

```python
_C.DATA_DIR = "/your/dataset/path"
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/wise-atta.git
   cd wise-atta
   ```

2. Create the conda environment:
   ```bash
   conda update conda
   conda env create -f environment.yml
   conda activate wise-atta
   ```

---

## Usage

### Example Run on ImageNet-C

```bash
cd classification
python test_time.py --cfg cfgs/imagenet_c/wiseatta.yaml MODEL.ADAPTATION wise
```

### Configuration

Modify parameters in the config files (e.g., `cfgs/imagenet_c/wiseatta.yaml`):

- `MODEL.ADAPTATION`: Choose the adaptation method
- `MODEL.ORACLE_NUM`: Number of samples to label per batch
- `MODEL.LABEL_RATIO`: Fraction of batches to label
- Other hyperparameters as needed

---

## Batch Selection Strategies

You can replace `wise` in `MODEL.ADAPTATION` with:

- `wise` — WISE-ATTA batch selection (utility-based)
- `uniform` — Uniform batch selection
- `random` — Random batch selection

---

## Results

Across **ImageNet-C** and natural distribution shift benchmarks (**ImageNet-R**, **ImageNet-K**, and **ImageNet-A**), WISE-ATTA achieves competitive or improved robustness while using substantially fewer labels.


## Results on Natural Distribution Shifts

| #Labels | Method | R<br>RN50 | R<br>ViT | K<br>RN50 | K<br>ViT | A<br>RN50 | A<br>ViT | Avg<br>RN50 | Avg<br>ViT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| - | TENT | 57.8 | 53.4 | 69.5 | 65.6 | 99.9 | 77.4 | 75.7 | 65.5 |
| - | CoTTA | 57.3 | 55.4 | 69.9 | 98.2 | 99.8 | 79.3 | 75.7 | 77.6 |
| - | SAR | 57.2 | 48.8 | 68.5 | 70.4 | 99.9 | 74.9 | 75.2 | 64.7 |
| - | ETA | 54.0 | 48.8 | 64.3 | 59.4 | 99.8 | 75.9 | 72.7 | 61.4 |
| - | CEMA† | 51.4 | 44.6 | 65.6 | 60.0 | 97.7 | 72.9 | 71.6 | 59.2 |
| 3 | SimATTA | 51.3 | 45.1 | 64.0 | 57.2 | 97.2 | 72.4 | 70.8 | 58.2 |
| 3 | HILTTA | 52.6 | 43.9 | 63.3 | 58.1 | 98.3 | 72.2 | 71.4 | 58.1 |
| 1 | EATTA | 52.8 | 44.3 | 64.1 | 58.2 | 99.1 | 71.5 | 72.0 | 58.0 |
| 1 | **WISE-ATTA** | **51.2** | **42.6** | **63.8** | **57.2** | **97.7** | **69.9** | **70.9** | **56.6** |
| 0.5 | **WISE-ATTA** | 52.2 | 43.5 | 64.6 | 58.0 | 98.3 | 70.9 | 71.7 | 57.5 |


<p align="center">
  <img src="label_ratio.png" alt="Label Ratio Comparison" width="600"/>
</p>

---

## Acknowledgements

This repository is built upon the excellent codebases:

- [EATTA](https://github.com/flash1803/EATTA/)
- [test-time-adaptation](https://github.com/mariodoebler/test-time-adaptation)

We also thank the authors of prior TTA and ATTA methods that inspired this work, including TENT, SimATTA, CEMA, HILTTA, and EATTA.

---

## Contact

For questions or collaborations, please contact:

Muhammad Huzaifa  
muhammad.huzaifa [at] cispa.de

---

## Citation

If you find this work useful, please cite:

```bibtex

```
