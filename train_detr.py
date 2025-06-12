import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from models.detr import DETRModel
from utils.dataset import SoccerNetMVFoulsDataset  # assuming your dataset class
from utils.preprocessing import VideoTransform
import torch.optim as optim
import time
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    transform = VideoTransform()
    train_dataset = SoccerNetMVFoulsDataset(root_dir='data', split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    val_dataset = SoccerNetMVFoulsDataset(root_dir='data', split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    # Model
    model = DETRModel(num_classes=5)  # 4 + background
    model.to(device)

    # Optimizer and LR Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 10
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for images, targets in train_loader:
            # images: tuple of tensors, targets: tuple of dicts
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        lr_scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Time: {elapsed:.2f}s")

        # Save checkpoint if improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_detr_model.pth')
            print(f"Saved Best Model with loss {best_loss:.4f}")

    print("Training complete")

if __name__ == "__main__":
    train()

