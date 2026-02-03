
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## License

This dataset and accompanying materials are released under the  
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

You are free to use, share, and adapt the material for any purpose,  
provided that appropriate credit is given to the authors.


# mmHSense: Multi-Modal and Distributed mmWave ISAC Datasets for Human-Sensing

This repository provides a PyTorch-based framework for training and evaluating deep learning models on various mmWave ISAC datasets, including gesture recognition, pose estimation, localization, and gait identification. It supports classification and regression tasks, and includes options for background subtraction where applicable.

![overview_dataset_jeroen (2)_page-0001](https://github.com/user-attachments/assets/e848e512-4d0c-451b-b28e-bb54a9ada3d8)


## Dataset Download

You can download the mmHSense dataset from [IEEE DataPort](https://ieee-dataport.org/documents/mmhsense-multi-modal-and-distributed-mmwave-isac-datasets-human-sensing).

1. Download the ZIP file.
2. Extract all datasets.
3. Place the **files** in the root directory of this repository.


## Features
Supports multiple datasets:

**mmWGesture** – mmWave gesture recognition (classification)

**5GmmGesture** – 5G mmWave gesture recognition (classification)

**mmWPose** – mmWave skeletal pose estimation (regression)

**DISAC-mmVRPose** – VR-based mmWave pose estimation (regression)

**mmW-Loc** – mmWave localization with optional background subtraction (classification)

**mmW-GaitID** – mmWave gait identification with optional background subtraction (classification)

Generic ResNet18-based architecture for all datasets with flexible input channels.


Supports both classification and regression loss functions (CrossEntropyLoss and MSELoss).

## Configuration

All dataset options and hyperparameters are set via `config.yaml`.  
You can edit this file to choose your dataset, adjust training parameters, or enable optional features.
```yaml
Example `config.yaml`:
dataset: mmWGesture         # Options: mmWGesture, 5GmmGesture, mmWPose, DISAC-mmVRPose, mmW-Loc, mmW-GaitID
epochs: 10
batch_size: 32
lr: 0.001
background: false
```    
Usage: **python main.py**

## Citation

@article{bhat2025mmhsense,
  title={mmHSense: Multi-Modal and Distributed mmWave ISAC Datasets for Human Sensing},
  author={Bhat, Nabeel Nisar and Karnaukh, Maksim and Vandenbroeke, Stein and Lemoine, Wouter and Struye, Jakob and Lacruz, Jesus Omar and Kumar, Siddhartha and Moghaddam, Mohammad Hossein and Widmer, Joerg and Berkvens, Rafael and others},
  journal={arXiv preprint arXiv:2509.21396},
  year={2025}
}
[https://arxiv.org/abs/2509.21396](https://arxiv.org/abs/2509.21396)

## 📚 Research Using the mmHSense Dataset

The mmHSense dataset is actively used by recent state-of-the-art research to advance mmWave human sensing and Integrated Sensing and Communications (ISAC). These works demonstrate the dataset’s applicability across gesture recognition, gait analysis, and sensing–communication trade-off studies.

🚀 mmGAN: Semi-Supervised GAN for Improved Gesture Recognition in mmWave ISAC Systems

This work shows that semi-supervised GANs trained on mmHSense significantly improve gesture recognition accuracy, particularly when labeled data is scarce, highlighting the dataset’s value for data-efficient ISAC learning.
🔗 https://ieeexplore.ieee.org/document/11317966

🚶 Beyond Sub-6 GHz: mmWave Wi-Fi for Gait-Based Person Identification

Using mmHSense, this study explores gait-based person identification with mmWave Wi-Fi, demonstrating a privacy-preserving alternative to vision-based systems and emphasizing the potential of beyond-sub-6 GHz sensing.
🔗 https://arxiv.org/abs/2510.08160

⏱️ Millimeter-Wave Gesture Recognition in ISAC: Does Reducing Sensing Airtime Hamper Accuracy?

This paper investigates the impact of reducing sensing airtime on mmWave gesture recognition using power-per-beam-pair measurements from mmHSense. Our results show that reducing sensing to just 25% of the airtime leads to only a 0.15% drop in accuracy, demonstrating that mmWave ISAC can achieve high-quality sensing with minimal overhead. This approach preserves maximum communication throughput, making it ideal for wireless XR and other real-time applications.

Paper: https://arxiv.org/abs/2601.10733

Code: https://github.com/JakobStruye/isac-tradeoff

