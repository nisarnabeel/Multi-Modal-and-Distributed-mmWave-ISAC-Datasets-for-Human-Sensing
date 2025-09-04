import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torchvision.models as models
import os

# --- Generic dataset loader ---
def load_dataset(dataset_name, background_flag=True):
    """Loads datasets and handles classification/regression tasks"""
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
        data = torch.load("CSI_60GHz_data_user1.pth")        # [N, H, W]
        labels = torch.load("CSI_60GHz_labels_user1.pth")    # [N, 50, 25, 3]

        # Flatten spatial dimension of labels: 50*25*3=3750
        N = labels.shape[0]
        labels = labels.reshape(N, -1)                       # [N, 3750]
        num_classes = labels.shape[1]                        # 3750
        task_type = "regression"

    elif dataset_name == "DISAC-mmVRPose":
        data = torch.load("X_train_user1.pth")
        labels = torch.load("y_train_user1.pth")
        num_classes = 75
        task_type = "regression"

    elif dataset_name == "mmW-Loc":
        file_name = "60Ghz_X_y_combined_loc_with_background_subtraction.pth" if background_flag else "60Ghz_X_y_combined_loc_without_background_subtraction.pth"
        data_tuple = torch.load(file_name)
        if isinstance(data_tuple, tuple):
            data, labels = data_tuple
        else:
            data, labels = data_tuple, None
        num_classes = 20
        task_type = "classification"

    elif dataset_name == "mmW-GaitID":
        file_name = "60Ghz_X_y_combined_ID_with_background_subtraction.pth" if background_flag else "60Ghz_X_y_combined_ID_without_background_subtraction.pth"
        data_tuple = torch.load(file_name)
        if isinstance(data_tuple, tuple):
            data, labels = data_tuple
        else:
            data, labels = data_tuple, None
        num_classes = 20
        task_type = "classification"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Convert labels to long for classification
    if task_type == "classification" and labels is not None:
        labels = labels.long()

    # Ensure all inputs are 4D: (N, C, H, W)
    if len(data.shape) == 2:
        data = data.unsqueeze(1).unsqueeze(-1)
    elif len(data.shape) == 3:
        data = data.unsqueeze(1)  # add channel dimension
    elif len(data.shape) == 1:
        data = data.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

    print(f"Loaded {dataset_name} with data shape {data.shape} and labels shape {labels.shape if labels is not None else None}")
    return data, labels, num_classes, task_type

# --- ResNet-based model ---
class GenericResNet(nn.Module):
    def __init__(self, input_channels, output_dim, task_type="classification"):
        super().__init__()
        self.task_type = task_type
        self.model = models.resnet18(weights=None)
        # Adjust first conv layer to input channels
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adjust output layer
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)

    def forward(self, x):
        return self.model(x)

# --- Training, validation, testing ---
def train_val_test(data, labels, output_dim, task_type="classification", epochs=10, batch_size=32, lr=1e-3):
    # Split train/val/test
    if labels is not None:
        X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    else:
        X_train, X_temp = train_test_split(data, test_size=0.3, random_state=42)
        X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
        y_train = y_val = y_test = None

    train_dataset = TensorDataset(X_train, y_train) if y_train is not None else TensorDataset(X_train)
    val_dataset = TensorDataset(X_val, y_val) if y_val is not None else TensorDataset(X_val)
    test_dataset = TensorDataset(X_test, y_test) if y_test is not None else TensorDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_channels = X_train.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GenericResNet(input_channels, output_dim, task_type=task_type).to(device)
    criterion = nn.CrossEntropyLoss() if task_type=="classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_score = float('-inf') if task_type=="classification" else float('inf')
    best_model_path = "best_model.pth"

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float() if task_type=="regression" else y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float() if task_type=="regression" else y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
                if task_type=="classification":
                    pred = output.argmax(dim=1)
                    correct += (pred == y_batch).sum().item()
                    total += y_batch.size(0)
        if task_type=="classification":
            val_score = correct/total
            print(f"Validation Accuracy: {val_score:.4f}")
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), best_model_path)
        else:
            val_score = val_loss/len(val_loader)
            print(f"Validation Loss: {val_score:.4f}")
            if val_score < best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), best_model_path)

    # Load best model for testing
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Testing
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float() if task_type=="regression" else y_batch.to(device)
            output = model(X_batch)
            test_loss += criterion(output, y_batch).item()
            if task_type=="classification":
                pred = output.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)
    if task_type=="classification":
        print(f"Test Accuracy: {correct/total:.4f}")
    else:
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["mmWGesture","mmWPose","5GmmGesture","DISAC-mmVRPose","mmW-Loc","mmW-GaitID"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--background", action="store_true", help="Enable background subtraction for mmW-Loc and mmW-GaitID")
    args = parser.parse_args()

    data, labels, output_dim, task_type = load_dataset(args.dataset, background_flag=args.background)
    train_val_test(data, labels, output_dim, task_type=task_type, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
