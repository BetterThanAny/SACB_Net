# SACB-Net: Spatial-awareness Convolutions for Medical Image Registration
The official implementation of SACB-Net [![CVPR](https://img.shields.io/badge/CVPR2025-68BC71.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_SACB-Net_Spatial-awareness_Convolutions_for_Medical_Image_Registration_CVPR_2025_paper.html)  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.19592) 

## Env

### Option A: AutoDL / RTX 5090 (PyTorch 2.8.0+cu128, Python 3.12)

The `requirements.txt` pins `torch==1.13.1` which is **incompatible** with RTX 5090 (requires CUDA 12.8+). If PyTorch is already installed on your machine, install only the remaining dependencies:

```bash
export OMP_NUM_THREADS=1

pip install einops==0.8.1 monai==1.5.2 timm==0.9.2 tensorboard==2.19.0 \
    "numpy>=1.26" "scipy>=1.12" "scikit-image>=0.22" "matplotlib>=3.8" \
    MedPy pystrum natsort kmeans_gpu==0.0.5
```

> **Note:** Python 3.10+ removed `collections.Sequence`. The fix (`collections.abc.Sequence`) is already applied in `dataset/trans.py` of this fork.

### Option B: Original setup (older GPU, CUDA ≤ 12.1)

```bash
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```
## Dataset
Thanks [@Junyu](https://github.com/junyuchen245) for [the preprocessed IXI data].

[Abdomen CT-CT](https://learn2reg.grand-challenge.org/Datasets/)
[LPBA](https://loni.usc.edu/research/atlases)

## Weights Download
[Google Drive](https://drive.google.com/drive/folders/1XW19iuyCyg3YGmCpLFGGFjdPFi73xxwh?usp=share_link).

## Citation
```bibtex
@InProceedings{Cheng_2025_CVPR,
    author    = {Cheng, Xinxing and Zhang, Tianyang and Lu, Wenqi and Meng, Qingjie and Frangi, Alejandro F. and Duan, Jinming},
    title     = {SACB-Net: Spatial-awareness Convolutions for Medical Image Registration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5227-5237}
}
```

## Acknowledgments
We sincerely acknowledge the [ModeT](https://github.com/ZAX130/SmileCode), [CANNet](https://github.com/Duanyll/CANConv) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) projects.
