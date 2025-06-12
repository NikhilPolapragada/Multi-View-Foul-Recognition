import torch
from torch.utils.data import DataLoader
from models.timesformer import TimeSformerModel
from utils.dataset import SoccerNetMVFoulsVideoDataset  # Your video dataset class for clips
from utils.preprocessing import VideoTransform
import torch.optim as optim
import torch.nn as nn
import time

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    transform = VideoTransform()
    train_dataset = SoccerNetMVFoulsVideoDataset(root_dir='data', split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    val_dataset = SoccerNetMVFoulsVideoDataset(root_dir='data', split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Model (Assuming 12 classes: 4 severity + 8 action types combined)
    model = TimeSformerModel(num_classes=12)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 10
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for videos, labels in train_loader:
            videos = videos.to(device)               # Shape: (B, C, T, H, W)
            labels = labels.to(device).float()      # Multi-label one-hot vector

            optimizer.zero_grad()
            outputs = model(videos)                  # Shape: (B, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        lr_scheduler.step()
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device).float()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Time: {elapsed:.2f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_timesformer_model.pth')
            print(f"Saved Best Model with Val Loss {best_val_loss:.4f}")

    print("Training complete")

if __name__ == "__main__":
    train()

