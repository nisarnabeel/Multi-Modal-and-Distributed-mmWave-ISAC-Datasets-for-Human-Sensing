import os
import torch

def load_dataset(dataset_name, data_dir=".", background_flag=True):
    """Load datasets and preprocess them for training/testing."""

    def p(subfolder, filename):
        return os.path.join(data_dir, subfolder, filename)

    if dataset_name == "mmWGesture":
        data   = torch.load(p("mmWGesture", "BeamSNR_60GHz_data_ENV1.pth"),   weights_only=False)
        labels = torch.load(p("mmWGesture", "BeamSNR_60GHz_labels_ENV1.pth"), weights_only=False)
        num_classes = 10
        task_type   = "classification"

    elif dataset_name == "5GmmGesture":
        data   = torch.load(p("5GmmGesture", "PPBP_data_user1.pth"),  weights_only=False)
        labels = torch.load(p("5GmmGesture", "labels_user1.pth"),     weights_only=False)
        num_classes = 8
        task_type   = "classification"

    elif dataset_name == "mmWPose":
        data   = torch.load(p("mmWPose", "CSI_60GHz_data_user1.pth"),   weights_only=False)
        labels = torch.load(p("mmWPose", "CSI_60GHz_labels_user1.pth"), weights_only=False)
        N = labels.shape[0]
        labels      = labels.reshape(N, -1)
        num_classes = labels.shape[1]
        task_type   = "regression"

    elif dataset_name == "DISAC-mmVRPose":
        data   = torch.load(p("DISAC-mmVRPose", "X_train_user1"), weights_only=False)
        labels = torch.load(p("DISAC-mmVRPose", "y_train_user1"), weights_only=False)
        num_classes = labels.shape[1] if labels.ndim > 1 else int(labels.max().item()) + 1
        task_type   = "regression"

    elif dataset_name == "mmW-Loc":
        fname = ("60GHz_X_y_combined_loc_with_background_subtraction.pth"
                 if background_flag else
                 "60GHz_X_y_combined_loc_without_background_subtraction.pth")
        data_tuple  = torch.load(p("mmW-Loc", fname), weights_only=False)
        data, labels = data_tuple if isinstance(data_tuple, tuple) else (data_tuple, None)
        num_classes = 20
        task_type   = "classification"

    elif dataset_name == "mmW-GaitID":
        fname = ("60GHz_X_y_combined_GaitID_with_background_subtraction.pth"
                 if background_flag else
                 "60GHz_X_y_combined_GaitID_without_background_subtraction.pth")
        data_tuple  = torch.load(p("mmW-GaitID", fname), weights_only=False)
        data, labels = data_tuple if isinstance(data_tuple, tuple) else (data_tuple, None)
        num_classes = 20
        task_type   = "classification"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if task_type == "classification" and labels is not None:
        labels = labels.long()

    # Ensure data is 4D: (N, C, H, W)
    if len(data.shape) == 2:
        data = data.unsqueeze(1).unsqueeze(-1)
    elif len(data.shape) == 3:
        data = data.unsqueeze(1)
    elif len(data.shape) == 1:
        data = data.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

    print(f"Loaded {dataset_name}: data={data.shape}  labels={labels.shape if labels is not None else None}")
    return data, labels, num_classes, task_type
