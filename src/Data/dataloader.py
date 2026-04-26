import os
from src.Data import transform
from torchvision import datasets
from torch.utils.data import DataLoader

split_dir = "/teamspace/studios/this_studio/split_dataset"

train_dataset = datasets.ImageFolder(  #loads images from a folder structure
    os.path.join(split_dir, "train"),
    transform=transform.get_transform()
)

test_dataset  = datasets.ImageFolder(
    os.path.join(split_dir, "test"),
    transform=transform.get_transform()
)

val_dataset   = datasets.ImageFolder(
    os.path.join(split_dir, "val"),
    transform=transform.get_transform()
)


train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True,  num_workers=4,
                          pin_memory=True)

test_loader  = DataLoader(test_dataset,  batch_size=32,
                          shuffle=False, num_workers=4,
                          pin_memory=True)

val_loader   = DataLoader(val_dataset,   batch_size=32,
                          shuffle=False, num_workers=4,
                          pin_memory=True)



if __name__ == '__main__':
    print(f"Classes  : {len(train_dataset.classes)}")
    print(f"Train    : {len(train_dataset)}")
    print(f"Val      : {len(val_dataset)}")
    print(f"Test     : {len(test_dataset)}")
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)
    print(labels.tolist())