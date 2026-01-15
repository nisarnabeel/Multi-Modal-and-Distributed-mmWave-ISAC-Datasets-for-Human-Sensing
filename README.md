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

Example `config.yaml`:
dataset: mmWGesture         # Options: mmWGesture, 5GmmGesture, mmWPose, DISAC-mmVRPose, mmW-Loc, mmW-GaitID
epochs: 10
batch_size: 32
lr: 0.001
background: false           # Only for mmW-Loc and mmW-GaitID

## Citation

@misc{bhat2025mmhsensemultimodaldistributedmmwave,
      title={mmHSense: Multi-Modal and Distributed mmWave ISAC Datasets for Human Sensing}, 
      author={Nabeel Nisar Bhat and Maksim Karnaukh and Stein Vandenbroeke and Wouter Lemoine and Jakob Struye and Jesus Omar Lacruz and Siddhartha Kumar and Mohammad Hossein Moghaddam and Joerg Widmer and Rafael Berkvens and Jeroen Famaey},
      year={2025},
      eprint={2509.21396},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2509.21396](https://arxiv.org/abs/2509.21396)}, 
}
## 📚 Research Using the mmHSense Dataset
The mmHSense dataset has been adopted by recent state-of-the-art works to advance human sensing at mmWave frequencies:

mmGAN: Semi-Supervised GAN for Improved Gesture Recognition in mmWave ISAC Systems This work leverages mmHSense to demonstrate how semi-supervised GAN-based learning significantly improves gesture recognition performance under limited labeled data conditions. - https://ieeexplore.ieee.org/document/11317966


Beyond Sub-6 GHz: Leveraging mmWave Wi-Fi for Gait-Based Person Identification This study utilizes mmHSense to explore gait-based person identification using mmWave Wi-Fi, highlighting the potential of beyond–sub-6 GHz sensing for privacy-preserving person identification. - https://arxiv.org/abs/2510.08160

