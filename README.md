# DMLCDNet: Multi-source Landslide Change Detection with Sentinel-2, Deformation, and Topographic Information

Official implementation of **DMLCDNet** for landslide change detection by integrating multispectral, deformation, and topographic information.

- **Paper (IJAEOG 2026):** https://doi.org/10.1016/j.jag.2026.105116  
- **Dataset:** https://pan.baidu.com/s/1IQ3ZYwwEgS3uSeVxj8qx5w?pwd=ax4k (code: `ax4k`)  
- **Contact:** Bo Liu (`trifurs@whu.edu.cn`)

---

## Overview

DMLCDNet is a deep learning framework for **landslide change detection** in complex mountainous environments. The method uses **Sentinel-2 multispectral observations** as the primary remote sensing input and further integrates **deformation-related** and **topographic** information to improve discrimination between landslide and non-landslide change regions.

This repository is designed to support research and reproduction for scenarios such as:

- post-earthquake or heavy-rainfall landslide mapping,
- rapid disaster response and damage assessment,
- landslide inventory update,
- remote sensing based geohazard monitoring and risk analysis.

---

## Paper Information

Liu, B., Li, D., Xiao, X., Shao, Z., Li, Y., Zhang, H., Chen, Y., Zhang, Z., Wang, S., & Chang, B. (2026). Enhancing Sentinel-2 landslide change detection by integrating multispectral, deformation, and topographic information. *International Journal of Applied Earth Observation and Geoinformation*, 146, 105116. https://doi.org/10.1016/j.jag.2026.105116

Paper: https://doi.org/10.1016/j.jag.2026.105116

---

## Method

According to the codebase and paper scope, DMLCDNet is built for binary landslide change extraction with multi-source feature fusion:

- **Multispectral information** (from Sentinel-2 style optical observations) provides land-cover and spectral change cues.
- **Deformation information** contributes surface movement cues related to landslide activity.
- **Topographic information** provides terrain constraints and context (elevation/slope/geomorphic background).

In implementation, training and prediction scripts dynamically load the model definition and split inputs into:

- pre-event image features,
- post-event image features,
- dynamic multi-source auxiliary features.

Please refer to the paper for detailed architectural design and full experimental analysis. If you need exact block-level network interpretation, use the repository model files and corresponding configuration XML files as the authoritative source.

---


## Model Architecture (from the Paper)

The model structure figure from the original paper is shown below:

![DMLCDNet framework (from Liu et al., 2026)](https://ars.els-cdn.com/content/image/1-s2.0-S1569843226000324-gr1.jpg)

Source paper: https://doi.org/10.1016/j.jag.2026.105116

---

## Repository Structure

```text
DMLCDNet/
├── README.md
├── LICENSE
└── Code/
    ├── train.py                     # training entry
    ├── test.py                      # testing/evaluation entry
    ├── utils.py                     # metrics, evaluation, helper functions
    ├── configs/
    │   ├── __base_param__.xml       # shared/default configuration
    │   ├── dmlcdnet.xml             # DMLCDNet training configuration
    │   └── comparison/*.xml         # configs for comparison models
    ├── dataset/
    │   └── landslide_dataset.py     # dataset loading/splitting/cache logic
    ├── model/
    │   ├── dmlcdnet.py              # DMLCDNet model definition
    │   └── comparison/*.py          # baseline comparison models
    ├── loss/
    │   ├── loss.py
    │   ├── dice_loss.py
    │   ├── focal_loss.py
    │   ├── boundary_loss.py
    │   └── tversky_loss.py
    ├── predict/
    │   ├── predict.py               # inference/prediction pipeline
    │   └── predict.xml              # prediction configuration
    └── tools/
        └── *.py                     # preprocessing/visualization utilities
```

---

## Installation

No `requirements.txt` / `environment.yml` is currently provided in the repository root.  
Please install dependencies according to imports used in the code.

A typical setup is:

```bash
conda create -n dmlcdnet python=3.10
conda activate dmlcdnet
pip install torch torchvision numpy tqdm pillow rasterio
```

> Note: adjust package versions according to your CUDA/driver/OS environment and your local reproducibility target.

---

## Dataset

The dataset used in this study can be downloaded from Baidu Netdisk:

https://pan.baidu.com/s/1IQ3ZYwwEgS3uSeVxj8qx5w?pwd=ax4k

Extraction code: `ax4k`

If you cannot access Baidu Netdisk, please contact the first author.

Bo Liu  
Email: trifurs@whu.edu.cn

After downloading the dataset, please organize files according to the dataset loader and config files in this repository.

From the current data loader logic, dataset samples are expected to include:

- multi-source feature rasters under `data/<feature_type>/...`,
- landslide change labels under `label/...` (TIFF format by default),
- optional `label_png/...` when PNG labels are explicitly enabled.

---

## Data Organization

Based on `Code/dataset/landslide_dataset.py`, the expected root contains `data` and `label` folders, where each feature type is a subfolder under `data`.

Example (adapt to your own dataset path):

```text
<your_data_root>/
├── data/
│   ├── feature_type_1/
│   │   ├── sample_a.tif
│   │   └── ...
│   ├── feature_type_2/
│   │   ├── sample_a.tif
│   │   └── ...
│   └── ...
├── label/
│   ├── sample_a.tif
│   └── ...
└── (optional) label_png/
    ├── sample_a.png
    └── ...
```

Important notes:

- Sample IDs are derived from label filenames.
- For each sample ID, corresponding TIFF files should exist in every enabled feature subfolder.
- Dataset split is handled inside the loader (`train/val/test`) by split ratios and random seed.
- Paths are configured through XML config files (e.g., `Code/configs/dmlcdnet.xml`, `Code/predict/predict.xml`).

---

## Training

Train with the provided training script and configuration:

```bash
cd Code
python train.py --config configs/dmlcdnet.xml
```

You may modify data/checkpoint/log paths in the XML configuration according to your local environment.

---

## Validation / Testing

Run evaluation/testing using:

```bash
cd Code
python test.py --config configs/dmlcdnet.xml
```

Metrics reported in the training/testing pipeline include (based on current scripts and utility functions):

- Precision
- Recall
- F1 score
- Slide IoU
- Mean IoU

---

## Inference / Prediction

Use the prediction script with its XML config:

```bash
cd Code/predict
python predict.py --config predict.xml
```

For inference, prepare the same multi-source input format as training (including paired temporal optical inputs and auxiliary dynamic features expected by the model/data pipeline), and set checkpoint/data/output paths in config files before running.

---

## Pretrained Weights

Pretrained weights are not included in this repository. Users can train the model using the provided scripts and their prepared dataset.

---

## Results

For detailed experimental results, ablation studies, and comparisons with other methods, please refer to the original paper.

Please refer to the paper for the detailed network architecture figures and visual result demonstrations.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{liu2026dmlcdnet,
  title={Enhancing Sentinel-2 landslide change detection by integrating multispectral, deformation, and topographic information},
  author={Liu, Bo and Li, D. and Xiao, X. and Shao, Z. and Li, Y. and Zhang, H. and Chen, Y. and Zhang, Z. and Wang, S. and Chang, B.},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={146},
  pages={105116},
  year={2026},
  doi={10.1016/j.jag.2026.105116}
}
```

---

## Contact

For questions about the dataset, code, or paper, please contact:

Bo Liu  
Email: trifurs@whu.edu.cn

---

## Acknowledgements

We thank the open-source communities in remote sensing, geospatial data processing, and deep learning (especially the PyTorch ecosystem) for foundational tools and libraries.

---

## License

This repository includes a `LICENSE` file. Please check `LICENSE` for the detailed terms before use.
