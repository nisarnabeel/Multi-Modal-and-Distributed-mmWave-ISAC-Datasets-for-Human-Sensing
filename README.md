> **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to use, share, and adapt with attribution.

# mmHSense: Multi-Modal and Distributed mmWave ISAC Datasets for Human Sensing

A PyTorch framework for training and evaluating deep learning models on mmWave ISAC datasets, covering gesture recognition, pose estimation, localization, and gait identification.

![Dataset overview](https://github.com/user-attachments/assets/e848e512-4d0c-451b-b28e-bb54a9ada3d8)

📄 [IEEE Access Paper](https://ieeexplore.ieee.org/document/11511681) · 📦 [Download Dataset (IEEE DataPort)](https://ieee-dataport.org/documents/mmhsense-multi-modal-and-distributed-mmwave-isac-datasets-human-sensing)

---

## Datasets

| Name | Task | Type |
|------|------|------|
| **mmWGesture** | mmWave gesture recognition | Classification |
| **5GmmGesture** | 5G mmWave gesture recognition | Classification |
| **mmWPose** | mmWave skeletal pose estimation | Regression |
| **DISAC-mmVRPose** | VR-based mmWave pose estimation | Regression |
| **mmW-Loc** | mmWave localization (+ optional background subtraction) | Classification |
| **mmW-GaitID** | mmWave gait identification (+ optional background subtraction) | Classification |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download the dataset from [IEEE DataPort](https://ieee-dataport.org/documents/mmhsense-multi-modal-and-distributed-mmwave-isac-datasets-human-sensing) and extract it. The data is organized into per-dataset subfolders:

```
/path/to/mmhsense/
├── mmWGesture/
│   ├── BeamSNR_60GHz_data_ENV1.pth
│   └── BeamSNR_60GHz_labels_ENV1.pth
├── 5GmmGesture/
│   ├── PPBP_data_user1.pth
│   └── labels_user1.pth
├── mmWPose/
│   ├── CSI_60GHz_data_user1.pth
│   └── CSI_60GHz_labels_user1.pth
├── DISAC-mmVRPose/
│   ├── X_train_user1
│   └── y_train_user1
├── mmW-Loc/
│   ├── 60GHz_X_y_combined_loc_with_background_subtraction.pth
│   └── 60GHz_X_y_combined_loc_without_background_subtraction.pth
└── mmW-GaitID/
    ├── 60GHz_X_y_combined_GaitID_with_background_subtraction.pth
    └── 60GHz_X_y_combined_GaitID_without_background_subtraction.pth
```

### 3. Configure

Edit `config.yaml` to select your dataset and point to the data directory:

```yaml
dataset: mmWGesture   # mmWGesture | 5GmmGesture | mmWPose | DISAC-mmVRPose | mmW-Loc | mmW-GaitID
data_dir: /path/to/mmhsense  # folder containing the dataset subfolders above
epochs: 10
batch_size: 32
lr: 0.001
background: false     # Enable background subtraction (mmW-Loc and mmW-GaitID only)
```

### 4. Train

```bash
python main.py
```

Or point to a custom config file:

```bash
python main.py --config config.yaml
```

---

## Features

- **Generic ResNet18 backbone** with flexible input channels, supporting all datasets out of the box
- **Classification and regression** tasks via `CrossEntropyLoss` and `MSELoss`
- **Stratified 80/10/10 train/val/test split** with best-val-accuracy checkpointing
- **Optional background subtraction** for localization and gait datasets

---

## Citation

If you use mmHSense in your research, please cite:

```bibtex
@ARTICLE{11511681,
  author    = {Bhat, Nabeel Nisar and Karnaukh, Maksim and Vandenbroeke, Stein and Lemoine, Wouter
               and Struye, Jakob and Kumar, Siddhartha and Moghaddam, Mohammad Hossein
               and Lacruz, Jesus Omar and Widmer, Joerg and Berkvens, Rafael and Famaey, Jeroen},
  journal   = {IEEE Access},
  title     = {mmHSense: Multi-Modal and Distributed mmWave ISAC Datasets for Human Sensing},
  year      = {2026},
  volume    = {14},
  pages     = {69961--69971},
  doi       = {10.1109/ACCESS.2026.3691174}
}
```

---

## Research Using mmHSense

The dataset is actively used by several recent works spanning gesture recognition, gait analysis, and sensing–communication trade-off studies.

**mmGAN: Semi-Supervised GAN for Improved Gesture Recognition in mmWave ISAC Systems**
Shows that semi-supervised GANs trained on mmHSense significantly improve gesture recognition accuracy when labeled data is scarce.
→ [IEEE Xplore](https://ieeexplore.ieee.org/document/11317966)

**Beyond Sub-6 GHz: mmWave Wi-Fi for Gait-Based Person Identification**
Explores gait-based person identification with mmWave Wi-Fi as a privacy-preserving alternative to vision-based systems.
→ [arXiv](https://arxiv.org/abs/2510.08160)   [Code](https://github.com/MaksimKarnaukh/MasterThesis_mmWavePI)

**Millimeter-Wave Gesture Recognition in ISAC: Does Reducing Sensing Airtime Hamper Accuracy?**
Demonstrates that reducing sensing to just 25% of airtime causes only a 0.15% accuracy drop — enabling high-quality sensing with minimal communication overhead, ideal for wireless XR applications.
→ [arXiv](https://arxiv.org/abs/2601.10733) · [Code](https://github.com/JakobStruye/isac-tradeoff)
