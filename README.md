
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

## Citation

@article{bhat2025mmhsense,
  title={mmHSense: Multi-Modal and Distributed mmWave ISAC Datasets for Human Sensing},
  author={Bhat, Nabeel Nisar and Karnaukh, Maksim and Vandenbroeke, Stein and Lemoine, Wouter and Struye, Jakob and Lacruz, Jesus Omar and Kumar, Siddhartha and Moghaddam, Mohammad Hossein and Widmer, Joerg and Berkvens, Rafael and others},
  journal={arXiv preprint arXiv:2509.21396},
  year={2025}
}
https://arxiv.org/abs/2509.21396](https://arxiv.org/abs/2509.21396)

## 📚 Research Using the mmHSense Dataset
The mmHSense dataset has been adopted by recent state-of-the-art works to advance human sensing at mmWave frequencies:

mmGAN: Semi-Supervised GAN for Improved Gesture Recognition in mmWave ISAC Systems This work leverages mmHSense to demonstrate how semi-supervised GAN-based learning significantly improves gesture recognition performance under limited labeled data conditions. - https://ieeexplore.ieee.org/document/11317966


Beyond Sub-6 GHz: Leveraging mmWave Wi-Fi for Gait-Based Person Identification This study utilizes mmHSense to explore gait-based person identification using mmWave Wi-Fi, highlighting the potential of beyond–sub-6 GHz sensing for privacy-preserving person identification. - https://arxiv.org/abs/2510.08160

Millimeter-Wave Gesture Recognition in ISAC: Does Reducing Sensing Airtime Hamper Accuracy? - Most Integrated Sensing and Communications (ISAC) systems require dividing airtime across their two modes. However, the specific impact of this decision on sensing performance remains unclear and underexplored. In this paper, we therefore investigate the impact on a gesture recognition system using a Millimeter-Wave (mmWave) ISAC system. With our dataset of power per beam pair gathered with two mmWave devices performing constant beam sweeps while test subjects performed distinct gestures, we train a gesture classifier using Convolutional Neural Networks. We then subsample these measurements, emulating reduced sensing airtime, showing that a sensing airtime of 25 % only reduces classification accuracy by 0.15 percentage points from full-time sensing. Alongside this high-quality sensing at low airtime, mmWave systems are known to provide extremely high data throughputs, making mmWave ISAC a prime enabler for applications such as truly wireless Extended Reality -  https://arxiv.org/abs/2601.10733

