import torch 
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def count_param(self) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad  # filters only trainable parameters
        )


    def summary(self):
        print(f'model: {self.__class__.__name__}')
        print(f'no of classes: {self.num_classes}')
        print(f'no of parameters: {self.count_param()}')


    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        print(f'model saved to : {path} successfully')

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        print(f'model loaded successfully from {path}')

