import shutil
import os

data_path1 = "/teamspace/studios/this_studio/dataset/datasets/Nischal/13_data_set"
data_path2 = "/teamspace/studios/this_studio/dataset/datasets/Nischal/20 data_set"

new_path = "/teamspace/studios/this_studio/dataset/merged_data"

# Create destination folder if it doesn't exist
os.makedirs(new_path, exist_ok=True)

# Move all folders/files from both source paths
for source_path in [data_path1, data_path2]:
    for item in os.listdir(source_path):
        src = os.path.join(source_path, item)
        dst = os.path.join(new_path, item)

        # Handle duplicate folder/file names
        if os.path.exists(dst):
            print(f"Skipped (already exists): {item}")
            continue

        shutil.move(src, dst)
        print(f"Moved: {item}")

