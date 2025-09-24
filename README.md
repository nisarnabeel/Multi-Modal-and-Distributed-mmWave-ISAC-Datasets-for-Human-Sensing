# Multi-Modal-and-Distributed-mmWave-ISAC-Datasets-for-Human-Sensing

**mmWave ISAC Datasets: mmHSense**

This repository provides a PyTorch-based framework for training and evaluating deep learning models on various mmWave ISAC datasets, including gesture recognition, pose estimation, localization, and gait identification. It supports classification and regression tasks, and includes options for background subtraction where applicable.

![overview_dataset_jeroen (2)_page-0001](https://github.com/user-attachments/assets/e848e512-4d0c-451b-b28e-bb54a9ada3d8)


## Dataset Download

You can download the mmHSense dataset from [IEEE DataPort](https://ieee-dataport.org/documents/mmhsense-multi-modal-and-distributed-mmwave-isac-datasets-human-sensing).

1. Download the ZIP file.
2. Extract all datasets.
3. Place the extracted folders and place the files in the root directory of this repository.


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
```yaml
dataset: mmWGesture        # Options: mmWGesture, 5GmmGesture, mmWPose, DISAC-mmVRPose, mmW-Loc, mmW-GaitID
epochs: 10
batch_size: 32
lr: 0.001
background: false           # Only for mmW-Loc and mmW-GaitID


Usage:python main.py
