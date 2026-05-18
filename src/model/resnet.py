import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights
from src.model.base_model import BaseModel 
from src.data.dataloader import num_classes

class ResNetClassifier(BaseModel):
    def __init__(self, n_classes: int, pretrained: bool):
        super().__init__(n_classes)
        self.n_classes = n_classes
        self.model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

if __name__ == "__main__":

    model = ResNetClassifier(
        n_classes = num_classes,
        pretrained  = True,
    )

    model.summary()

  
    dummy_input = torch.randn(32, 3, 224, 224)
    output      = model(dummy_input)

    print(f"Forward pass successful!")
    print(f"Input  shape : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")













