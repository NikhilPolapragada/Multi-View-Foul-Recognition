import torch
import torchvision
from torchvision.models.detection import detr_resnet50
from torchvision.models.detection.transform import GeneralizedRCNNTransform

class DETRModel(torch.nn.Module):
    def __init__(self, num_classes=5):  # 4 classes + background
        super(DETRModel, self).__init__()
        # Load pretrained DETR model from torchvision
        self.model = detr_resnet50(pretrained=True, num_classes=num_classes)
        
        # Optional: modify the transform if you want custom input size
        self.model.transform = GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, images, targets=None):
        """
        images: list of tensors [C, H, W]
        targets: Optional list of dicts with 'boxes' and 'labels' for training
        """
        return self.model(images, targets)

if __name__ == "__main__":
    # Quick test
    model = DETRModel(num_classes=5)
    model.eval()
    dummy_image = torch.rand(3, 800, 800)
    output = model([dummy_image])
    print(output)

