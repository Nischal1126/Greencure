import os
import shutil
import random

source_dir = "/teamspace/studios/this_studio/dataset/merged_data"
destination_dir = "/teamspace/studios/this_studio/split_dataset"

train_ratio = 0.7
test_ratio = 0.15
val_ratio = 0.15

def split_dataset(source_dir, destination_dir, train_ratio, test_ratio, val_ratio):
    random.seed(42)

    for split in ["train", "test", "val"]:
        os.makedirs(os.path.join(destination_dir, split), exist_ok = True)

    classes = [c for c in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, c))]

    print(f'total no of classes: {len(classes)}')

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        test_end = train_end + int(total * test_ratio)
        
        splits = {
            "train" : images[:train_end],
            "test"  : images[train_end:test_end],
            "val"   : images[test_end:]
            }

        for split, files in splits.items():
            dest = os.path.join(destination_dir, split, cls)
            os.makedirs(dest, exist_ok= True)
            for f in files:
                shutil.copy2(os.path.join(cls_path, f), os.path.join(dest, f))


split_dataset(source_dir, destination_dir, train_ratio, test_ratio, val_ratio)
        
