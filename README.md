# WISE-ATTA

Official PyTorch implementation of  
**WISE-ATTA: When to Ask for Labels in Budgeted Active Test-Time Adaptation**

> Budget-aware active test-time adaptation for long test streams with limited supervision.

<p align="center">
  <img src="frameworks.png" alt="Framework" width="500"/>
</p>

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


Across **ImageNet-C** and natural distribution shift benchmarks (**ImageNet-R**, **ImageNet-K**, and **ImageNet-A**), WISE-ATTA achieves competitive or improved robustness while using substantially fewer labels.

<p align="center">
  <img src="label_ratio.png" alt="Framework" width="600"/>
</p>

## Environment

Tested with:

- **Driver Version:** 550.67  
- **CUDA Version:** 12.4  
- **Python Version:** 3.10.12  

To use our repository, create conda environment:

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


Example Run on ImageNet-C
cd classification
python test_time.py --cfg cfgs/imagenet_c/wiseatta.yaml MODEL.ADAPTATION wise
Batch Selection Strategies

You can replace wise in MODEL.ADAPTATION with:

wise — WISE-ATTA batch selection
uniform — uniform batch selection
random — random batch selection



Acknowledgements

This repository is built upon the excellent codebases:

EATTA
test-time-adaptation

We also thank the authors of prior TTA and ATTA methods that inspired this work, including TENT, SimATTA, CEMA, HILTTA, and EATTA.

Contact

For questions or collaborations, please contact:

Muhammad Huzaifa
muhammad.huzaifa [at] cispa.de
