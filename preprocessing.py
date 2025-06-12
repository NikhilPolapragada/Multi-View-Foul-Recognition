import torchvision.transforms as transforms
import torch

class VideoTransform:
    def __init__(self, resize=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Compose video transformations:
        - Resize frames
        - Convert to tensor
        - Normalize
        """
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, frames):
        # frames: numpy array (T, H, W, C)
        # Apply transform to each frame, then stack
        transformed = [self.transform(frame) for frame in frames]
        # Stack frames to tensor of shape (T, C, H, W)
        video_tensor = torch.stack(transformed)
        return video_tensor

