import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class SoccerNetMVFoulsDataset(Dataset):
    """
    PyTorch Dataset for SoccerNet Multi-View Foul Recognition.

    Expects data folder structure:
    data/
      train/
        <video_clips>...
      val/
        <video_clips>...
    
    Each sample is a multi-view video with labels:
    - first_label: foul severity (0-3)
    - second_label: action type (0-7)
    """

    def __init__(self, root_dir, split='train', transform=None, max_frames=32):
        """
        Args:
            root_dir (str): Path to the SoccerNet mvfouls dataset root.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to apply on frames.
            max_frames (int): Max number of frames to sample from each video.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.max_frames = max_frames

        self.video_files = []
        self.labels = []  # list of tuples: (first_label, second_label)

        # Load labels from CSV file: train_labels.csv, val_labels.csv etc.
        import csv
        label_csv_path = os.path.join(root_dir, f"{split}_labels.csv")
        if not os.path.isfile(label_csv_path):
            raise FileNotFoundError(f"Label file not found: {label_csv_path}")

        with open(label_csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                video_name, first_label, second_label = row
                video_path = os.path.join(self.root_dir, video_name)
                if os.path.isfile(video_path):
                    self.video_files.append(video_path)
                    self.labels.append((int(first_label), int(second_label)))
                else:
                    print(f"Warning: Video file not found: {video_path}")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        first_label, second_label = self.labels[idx]

        frames = self._load_video_frames(video_path)

        if self.transform:
            frames = self.transform(frames)

        # Convert frames to tensor shape (C, T, H, W)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0

        return frames, torch.tensor(first_label), torch.tensor(second_label)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, frame_count - 1, self.max_frames).astype(int)

        frames = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()

        frames = np.array(frames)
        if frames.shape[0] < self.max_frames:
            last_frame = frames[-1]
            pad_count = self.max_frames - frames.shape[0]
            padding = np.repeat(last_frame[np.newaxis, :, :, :], pad_count, axis=0)
            frames = np.concatenate([frames, padding], axis=0)

        return frames

