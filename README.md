# WISE-ATTA

Official PyTorch implementation of **WISE-ATTA: When to Ask for Labels in Budgeted Active Test-Time Adaptation**, accepted at the **IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2027)**.

**Paper:** [arXiv:2609.37687](https://arxiv.org/abs/2609.37687)

---

## Environment

Experiments were conducted using the following environment:

- **NVIDIA Driver:** 550.163.01
- **CUDA:** 12.4
- **Python:** 3.10.12

## Installation

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

## Datasets

Experiments are conducted on the following benchmarks:

- [ImageNet-C](https://zenodo.org/records/2235448) — Common corruptions
- [ImageNet-R](https://github.com/hendrycks/imagenet-r) — Renditions
- [ImageNet-K (ImageNet-Sketch)](https://github.com/HaohanWang/ImageNet-Sketch) — Sketches
- [ImageNet-A](https://github.com/hendrycks/natural-adv-examples) — Adversarial examples

After downloading the datasets, specify the dataset directory in `classification/conf.py`:

```python
_C.DATA_DIR = "/your/dataset/path"
```

## Usage

To run WISE-ATTA on ImageNet-C using the default configuration:

```bash
cd classification
python test_time.py --cfg cfgs/imagenet_c/wiseatta.yaml MODEL.ADAPTATION wise
```

### Configuration

Experimental settings can be modified through the corresponding configuration files, such as `cfgs/imagenet_c/wiseatta.yaml`.

| Parameter | Description |
|---|---|
| `MODEL.ADAPTATION` | Batch selection strategy (`wise`, `uniform`, or `random`) |
| `MODEL.EDGE_ARCH` | Backbone architecture used for adaptation |
| `MODEL.LABEL_RATIO` | Fraction of test batches selected for labeling |

## Results

The following table reports classification error (%) on ImageNet-R, ImageNet-K, and ImageNet-A under different labeling budgets. Lower values indicate better performance.

| # Labels | Method | ImageNet-R | ImageNet-K | ImageNet-A | Avg. Error |
|---|---|---:|---:|---:|---:|
| – | TENT | 57.8 | 69.5 | 99.9 | 75.7 |
| – | CoTTA | 57.3 | 69.9 | 99.8 | 75.7 |
| – | SAR | 57.2 | 68.5 | 99.9 | 75.2 |
| – | ETA | 54.0 | 64.3 | 99.8 | 72.7 |
| – | CEMA† | 51.4 | 65.6 | 97.7 | 71.6 |
| 3 | SimATTA | 51.3 | 64.0 | 97.2 | 70.8 |
| 3 | HILTTA | 52.6 | 63.3 | 98.3 | 71.4 |
| 1 | EATTA | 52.8 | 64.1 | 99.1 | 72.0 |
| 1 | **WISE-ATTA** | **51.2** | **63.8** | **97.7** | **70.9** |
| 0.5 | **WISE-ATTA** | **52.2** | **64.6** | **98.3** | **71.7** |

WISE-ATTA achieves an average error of **70.9%** with one label per batch and **71.7%** with an average budget of 0.5 labels per batch, demonstrating competitive performance under limited supervision.

## Acknowledgements

This implementation builds upon the following open-source repositories:

- [EATTA](https://github.com/flash1803/EATTA)
- [Test-Time Adaptation](https://github.com/mariodoebler/test-time-adaptation)

We acknowledge the contributions of prior work on test-time adaptation and active test-time adaptation, including TENT, SimATTA, CEMA, HILTTA, and EATTA.

## Citation

If you find this repository useful, please cite our WACV 2027 paper:

```bibtex
@inproceedings{huzaifa2027wiseatta,
  title     = {{WISE-ATTA}: When to Ask for Labels in Budgeted Active Test-Time Adaptation},
  author    = {Huzaifa, Muhammad and Schönherr, Lea and Eisenhofer, Thorsten},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

The preprint is available at [arXiv:2609.37687](https://arxiv.org/abs/2609.37687).

## Contact

For questions regarding the implementation or experimental results, please contact:

**Muhammad Huzaifa**  
CISPA Helmholtz Center for Information Security  
muhammad.huzaifa [at] cispa.de
