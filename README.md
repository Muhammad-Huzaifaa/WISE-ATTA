<div align="center">

# WISE-ATTA

### When to Ask for Labels in Budgeted Active Test-Time Adaptation

**Muhammad Huzaifa · Lea Schönherr · Thorsten Eisenhofer**

CISPA Helmholtz Center for Information Security

[![WACV 2027](https://img.shields.io/badge/WACV-2027-1f6feb?style=for-the-badge)](https://wacv.thecvf.com/Conferences/2027/index.html)
[![arXiv](https://img.shields.io/badge/arXiv-2609.37687-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.37687)
[![Hugging Face Paper](https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-FFD21E?style=for-the-badge)](https://huggingface.co/papers/2609.37687)

Official PyTorch implementation of the WACV 2027 paper
**[WISE-ATTA: When to Ask for Labels in Budgeted Active Test-Time Adaptation](https://arxiv.org/abs/2609.37687)**

</div>

<!-- Optional: add a teaser/overview figure
<p align="center">
  <img src="assets/teaser.png" width="85%" alt="WISE-ATTA overview">
</p>
-->

---

## 📑 Table of Contents

- [Environment](#️-environment)
- [Installation](#-installation)
- [Datasets](#-datasets)
- [Usage](#-usage)
- [Results](#-results)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🖥️ Environment

All experiments were conducted with the following setup:

| Component     | Version    |
|---------------|------------|
| NVIDIA Driver | 550.163.01 |
| CUDA          | 12.4       |
| Python        | 3.10.12    |

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Muhammad-Huzaifaa/WISE-ATTA.git
cd WISE-ATTA
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate wise-atta
```

## 🗂️ Datasets

We evaluate on the following distribution-shift benchmarks:

| Dataset | Shift Type | Source |
|---|---|---|
| ImageNet-C | Common corruptions | [Zenodo](https://zenodo.org/records/2235448) |
| ImageNet-R | Renditions | [GitHub](https://github.com/hendrycks/imagenet-r) |
| ImageNet-K (ImageNet-Sketch) | Sketches | [GitHub](https://github.com/HaohanWang/ImageNet-Sketch) |
| ImageNet-A | Natural adversarial examples | [GitHub](https://github.com/hendrycks/natural-adv-examples) |

After downloading, set the dataset root in `classification/conf.py`:

```python
_C.DATA_DIR = "/your/dataset/path"
```

## 🚀 Usage

Run WISE-ATTA on ImageNet-C with the default configuration:

```bash
cd classification
python test_time.py --cfg cfgs/imagenet_c/wiseatta.yaml MODEL.ADAPTATION wise
```

### Configuration

Experimental settings are controlled via YAML files (e.g., `cfgs/imagenet_c/wiseatta.yaml`) and can be overridden from the command line.

| Parameter | Description |
|---|---|
| `MODEL.ADAPTATION` | Batch selection strategy: `wise`, `uniform`, or `random` |
| `MODEL.EDGE_ARCH` | Backbone architecture used for adaptation |
| `MODEL.LABEL_RATIO` | Fraction of test batches selected for labeling |

## 📊 Results

Classification error (%, ↓) on ImageNet-R, ImageNet-K, and ImageNet-A under different labeling budgets (labels per batch). Best results within the ≤ 1 label budget are in **bold**.

| # Labels | Method | ImageNet-R | ImageNet-K | ImageNet-A | Avg. ↓ |
|:---:|---|:---:|:---:|:---:|:---:|
| – | TENT | 57.8 | 69.5 | 99.9 | 75.7 |
| – | CoTTA | 57.3 | 69.9 | 99.8 | 75.7 |
| – | SAR | 57.2 | 68.5 | 99.9 | 75.2 |
| – | ETA | 54.0 | 64.3 | 99.8 | 72.7 |
| – | CEMA† | 51.4 | 65.6 | 97.7 | 71.6 |
| 3 | SimATTA | 51.3 | 64.0 | 97.2 | 70.8 |
| 3 | HILTTA | 52.6 | 63.3 | 98.3 | 71.4 |
| 1 | EATTA | 52.8 | 64.1 | 99.1 | 72.0 |
| 1 | **WISE-ATTA (ours)** | **51.2** | **63.8** | **97.7** | **70.9** |
| 0.5 | **WISE-ATTA (ours)** | 52.2 | 64.6 | 98.3 | 71.7 |

> With only **one label per batch**, WISE-ATTA reaches **70.9%** average error, on par with methods using 3× more labels. With an average budget of **0.5 labels per batch**, it still achieves **71.7%**, outperforming EATTA at twice the budget.

## 🙏 Acknowledgements

This codebase builds upon the following open-source projects:

- [EATTA](https://github.com/flash1803/EATTA)
- [Test-Time Adaptation](https://github.com/mariodoebler/test-time-adaptation)

We also thank the authors of prior work on test-time and active test-time adaptation, including TENT, SimATTA, CEMA, HILTTA, and EATTA.

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{huzaifa2027wiseatta,
  title     = {{WISE-ATTA}: When to Ask for Labels in Budgeted Active Test-Time Adaptation},
  author    = {Huzaifa, Muhammad and Sch{\"o}nherr, Lea and Eisenhofer, Thorsten},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## 📬 Contact

For questions about the implementation or results, please open an [issue](https://github.com/Muhammad-Huzaifaa/WISE-ATTA/issues) or contact:

**Muhammad Huzaifa** · CISPA Helmholtz Center for Information Security · `muhammad.huzaifa [at] cispa.de`
