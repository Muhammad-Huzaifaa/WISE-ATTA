# 🍦EATTA🍦
This is the official project repository for **Effortless Active Labeling for Long-Term Test-Time Adaptation** by Guowei Wang and Changxing Ding* (**CVPR 2025**).

🍦EATTA🍦 relieves the increasing annotation burden in ATTA as online data scale grows by requiring only one labeled sample per batch, or even per multi-batches.

- 🍦EATTA🍦 identifies the most valuable sample in each batch for labeling from the perspective of single-step optimization.
- 🍦EATTA🍦 introduces a gradient norm-based debiasing method to balance the training objectives of the labeled and unlabeled data.

![Frameworks](EATTA-repo/frameworks.png)
---
# Environments
Driver Version: 550.67 | CUDA Version: 12.4 | Python Version：3.10.12

To use the repository, we provide a conda environment.
```
conda update conda
conda env create -f environment.yml
conda activate tta 
```
# Datasets
- [ImageNet-C](https://zenodo.org/records/2235448#.Yj2RO_co_mF)
- [ImageNet-R](https://github.com/hendrycks/imagenet-r)
- [ImageNet-K](https://github.com/HaohanWang/ImageNet-Sketch)
- [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
  
After downloading these datasets, you may need to change the path to the root directory in the file conf.py. For example,
```
# change path
_C.DATA_DIR = "/guowei/data" --> _C.DATA_DIR = "/usr/data"
```
# Baselines
- Baseline. We create an ATTA baseline based on [TENT](https://github.com/DequanWang/tent) (ICLR 2021). It andomly selects a specified number of samples from each online batch for manual/large model annotation, and perform adaptation using Eq.2.
- [SimATTA](https://github.com/divelab/ATTA) (ICLR 2024).
- [CEMA](https://github.com/chenyaofo/CEMA) (ICLR 2024).
- [HILTTA](https://github.com/Yushu-Li/HILTTA) (TMLR)
# Example
```
cd classification
CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/imagenet_c/eatta.yaml
```
Considering modifying the following parameters in './cfgs/imagenet_c/eatta.yaml' to test ATTA methods with different parameters. For example,
```
MODEL:
  ADAPTATION: eatta
  EDGE_CLOUD: True # No need to change
  CLOUD_ARCH: 'vit_l_16' # Architecture of the large model (annotator). Only works when 'LARGE_MODELS' is called.
  EDGE_ARCH: 'resnet50' # Architecture of the adapted model.
  EDGE_ARCH_WEIGHTS: 'IMAGENET1K_V1' # No need to change
  HUMAN_OR_LARGE_MODEL: 'HUMAN' # Switch the annotator: 'HUMAN'-->'LARGE_MODELS'.
  ORACLE_NUM: 1  # Change the number of samples for annotation in each batch: 1-->3.
  BUFFER: False # Not use or use the buffer: False-->True.
```


---
# Correspondence
Please contact Guowei Wang by eegw.wang [at] mail.scut.edu.cn

# Citation
If you find 🍦EATTA🍦 is helpful in your research, please considering citing our paper:
```
@inproceedings{wang2025effortless,
  title={Effortless active labeling for long-term test-time adaptation},
  author={Wang, Guowei and Ding, Changxing},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={25633--25642},
  year={2025}
}
```


# Acknowledgements
We appreciate the contribution of a brilliant codebase [https://github.com/mariodoebler/test-time-adaptation](https://github.com/mariodoebler/test-time-adaptation), which is developed by Robert A. Marsden and Mario Döbler. 
