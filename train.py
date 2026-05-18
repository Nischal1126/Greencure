import sys
sys.path.append("/teamspace/studios/this_studio")

from configs.config       import CONFIG
from src.data.dataloader  import (
    train_loader, val_loader,
    test_loader,  num_classes
)
from src.model.resnet     import ResNetClassifier
from src.training.trainer import Trainer

# Model
model = ResNetClassifier(
    n_classes  = num_classes,
    pretrained = CONFIG["model"]["pretrained"]
)
model.summary()

# Dataloaders
dataloaders = {
    "train" : train_loader,
    "val"   : val_loader,
    "test"  : test_loader
}

# Trainer
trainer = Trainer(
    model       = model,
    dataloaders = dataloaders,
    config      = CONFIG["training"]
)

trainer.train()