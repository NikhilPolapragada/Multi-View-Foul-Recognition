import torch
from torchvision.models.video import timesformer_swin_base_224

class TimeSformerModel(torch.nn.Module):
    def __init__(self, num_classes=12):  # 4 severity + 8 action types combined or multitask
        super(TimeSformerModel, self).__init__()
        # Load pretrained TimeSformer model (if available)
        self.model = timesformer_swin_base_224(pretrained=True)
        
        # Replace final layer with your classification head(s)
        self.model.head = torch.nn.Linear(self.model.head.in_features, num_classes)
    
    def forward(self, x):
        """
        x: input video tensor of shape (B, C, T, H, W)
        """
        return self.model(x)

if __name__ == "__main__":
    model = TimeSformerModel(num_classes=12)
    dummy_video = torch.randn(1, 3, 8, 224, 224)  # Batch 1, 3 channels, 8 frames
    out = model(dummy_video)
    print(out.shape)

