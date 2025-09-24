# Multi-Modal-and-Distributed-mmWave-ISAC-Datasets-for-Human-Sensing

**mmWave ISAC Datasets: mmHSense**

This repository provides a PyTorch-based framework for training and evaluating deep learning models on various mmWave ISAC datasets, including gesture recognition, pose estimation, localization, and gait identification. It supports classification and regression tasks, and includes options for background subtraction where applicable.

<iframe src="https://github.com/user-attachments/files/22512851/overview_dataset_jeroen.2.pdf" width="700" height="500"></iframe>


Supports multiple datasets:

**mmWGesture** – mmWave gesture recognition (classification)

**5GmmGesture** – 5G mmWave gesture recognition (classification)

**mmWPose** – mmWave skeletal pose estimation (regression)

**DISAC-mmVRPose** – VR-based mmWave pose estimation (regression)

**mmW-Loc** – mmWave localization with optional background subtraction (classification)

**mmW-GaitID** – mmWave gait identification with optional background subtraction (classification)

Generic ResNet18-based architecture for all datasets with flexible input channels.


Supports both classification and regression loss functions (CrossEntropyLoss and MSELoss).

Usage:python main.py
