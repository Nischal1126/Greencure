import os

root_dir = "/teamspace/studios/this_studio/split_dataset"
splits = ['train', 'val', 'test']

# Image extensions to count (ignores hidden files like .DS_Store)
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

print("=" * 65)
print(f"{'Class Name':<30} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
print("=" * 65)

# First, get a master list of all unique class folder names across the splits
all_classes = set()
for split in splits:
    split_path = os.path.join(root_dir, split)
    if os.path.exists(split_path):
        all_classes.update([
            d for d in os.listdir(split_path) 
            if os.path.isdir(os.path.join(split_path, d))
        ])

# Totals for the bottom summary
totals = {split: 0 for split in splits}

# Loop through each class and count files across train, val, and test
for class_name in sorted(all_classes):
    counts = {}
    for split in splits:
        class_folder_path = os.path.join(root_dir, split, class_name)
        
        if os.path.exists(class_folder_path):
            # Count only valid image files
            images = [
                f for f in os.listdir(class_folder_path)
                if f.lower().endswith(VALID_EXTENSIONS)
            ]
            img_count = len(images)
            counts[split] = img_count
            totals[split] += img_count
        else:
            counts[split] = 0
            
    print(f"{class_name:<30} | {counts['train']:<8} | {counts['val']:<8} | {counts['test']:<8}")

print("=" * 65)
print(f"{'TOTALS':<30} | {totals['train']:<8} | {totals['val']:<8} | {totals['test']:<8}")
print("=" * 65)