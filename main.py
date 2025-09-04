import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from models import GenericResNet

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
    model = GenericResNet(input_channels, output_dim).to(device)
    criterion = nn.CrossEntropyLoss() if task_type=="classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_score = float('-inf') if task_type=="classification" else float('inf')
    best_model_path = "best_model.pth"

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

        val_score = (correct/total if task_type=="classification" else val_loss/len(val_loader))
        print(f"Validation {'Accuracy' if task_type=='classification' else 'Loss'}: {val_score:.4f}")
        if (task_type=="classification" and val_score > best_val_score) or (task_type=="regression" and val_score < best_val_score):
            best_val_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # Load best model for testing
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

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

    print(f"Test {'Accuracy' if task_type=='classification' else 'Loss'}: {correct/total if task_type=='classification' else test_loss/len(test_loader):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # Load YAML config
    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data, labels, output_dim, task_type = load_dataset(cfg['dataset'], background_flag=cfg.get('background', False))
    train_val_test(
    data, labels, output_dim, task_type=task_type,
    epochs=cfg['epochs'], 
    batch_size=cfg['batch_size'], 
    lr=cfg['lr']
)