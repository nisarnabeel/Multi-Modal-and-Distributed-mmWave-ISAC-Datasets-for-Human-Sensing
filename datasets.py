import torch

def load_dataset(dataset_name, background_flag=True):
    """Load datasets and preprocess them for training/testing."""
    if dataset_name == "mmWGesture":
        data = torch.load("BeamSNR_60GHz_data_ENV1.pth")
        labels = torch.load("BeamSNR_60GHz_labels_ENV1.pth")
        num_classes = 10
        task_type = "classification"

    elif dataset_name == "5GmmGesture":
        data = torch.load("PPBP_data_user1.pth")
        labels = torch.load("labels_user1.pth")
        num_classes = 8
        task_type = "classification"

    elif dataset_name == "mmWPose":
        data = torch.load("CSI_60GHz_data_user1.pth")
        labels = torch.load("CSI_60GHz_labels_user1.pth")
        N = labels.shape[0]
        labels = labels.reshape(N, -1)  # Flatten spatial dims: [N, 50*25*3=3750]
        num_classes = labels.shape[1]
        task_type = "regression"

    elif dataset_name == "DISAC-mmVRPose":
        data = torch.load("X_train_user1.pth")
        labels = torch.load("y_train_user1.pth")
        num_classes = labels.shape[1] if labels.ndim > 1 else labels.max().item() + 1
        task_type = "regression"

    elif dataset_name == "mmW-Loc":
        file_name = "60Ghz_X_y_combined_loc_with_background_subtraction.pth" if background_flag else "60Ghz_X_y_combined_loc_without_background_subtraction.pth"
        data_tuple = torch.load(file_name)
        data, labels = data_tuple if isinstance(data_tuple, tuple) else (data_tuple, None)
        num_classes = 20
        task_type = "classification"

    elif dataset_name == "mmW-GaitID":
        file_name = "60Ghz_X_y_combined_ID_with_background_subtraction.pth" if background_flag else "60Ghz_X_y_combined_ID_without_background_subtraction.pth"
        data_tuple = torch.load(file_name)
        data, labels = data_tuple if isinstance(data_tuple, tuple) else (data_tuple, None)
        num_classes = 20
        task_type = "classification"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Convert classification labels to long
    if task_type == "classification" and labels is not None:
        labels = labels.long()

    # Ensure data is 4D: (N, C, H, W)
    if len(data.shape) == 2:
        data = data.unsqueeze(1).unsqueeze(-1)
    elif len(data.shape) == 3:
        data = data.unsqueeze(1)  # add channel dimension
    elif len(data.shape) == 1:
        data = data.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

    print(f"Loaded {dataset_name} with data shape {data.shape} and labels shape {labels.shape if labels is not None else None}")
    return data, labels, num_classes, task_type
