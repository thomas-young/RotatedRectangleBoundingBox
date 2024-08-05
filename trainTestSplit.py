from pathlib import Path 
import os
import pandas as pd
import shutil
import random


imagesPath = '/Users/tomyoung/Desktop/RotatedRectangleBoundingBox/Images'
labelsPath = '/Users/tomyoung/Desktop/RotatedRectangleBoundingBox/Labels'

def create_splits(base_path, split_ratios):
    # Ensure the base path exists
    base_path = Path(base_path)
    assert base_path.exists(), f"The directory {base_path} does not exist."

    # Create subdirectories for train, test, and val splits
    train_dir = base_path / 'train'
    test_dir = base_path / 'test'
    val_dir = base_path / 'val'
    
    for dir in [train_dir, test_dir, val_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Get all files in the base directory, assuming label files with .txt extension
    files = [f for f in os.listdir(base_path) if f.endswith('.txt')]
    random.shuffle(files)  # Shuffle files to randomize input

    # Split files according to specified ratios
    total_files = len(files)
    train_split = int(total_files * split_ratios['train'])
    test_split = int(total_files * split_ratios['test'])
    
    train_files = files[:train_split]
    test_files = files[train_split:train_split + test_split]
    val_files = files[train_split + test_split:]
    
    return train_files, test_files, val_files

def move_files(label_source_dir, label_dest_dir, files, images_source_base_path):
    subdirectory = label_dest_dir.name  # 'train', 'test', or 'val'
    for file in files:
        label_file_path = label_source_dir / file
        new_label_path = label_dest_dir / file
        shutil.move(str(label_file_path), str(new_label_path))
        
        # Corresponding image file handling
        image_file = file.replace('.txt', '.jpg')
        image_file_path = images_source_base_path / image_file
        print(image_file_path)
        new_image_path = images_source_base_path / subdirectory / image_file  # Correctly point to the subdirectory under 'Images'
        print(new_image_path)
        shutil.move(str(image_file_path), str(new_image_path))

# Define paths
images_path = Path('/Users/tomyoung/Desktop/RotatedRectangleBoundingBox/Images')
labels_path = Path('/Users/tomyoung/Desktop/RotatedRectangleBoundingBox/Labels')

# Create train, test, val subdirectories in both Images and Labels
create_splits(labels_path, {'train': 0.7, 'test': 0.15, 'val': 0.15})
create_splits(images_path, {'train': 0.7, 'test': 0.15, 'val': 0.15})

# Get the file splits
train_files, test_files, val_files = create_splits(labels_path, {'train': 0.7, 'test': 0.15, 'val': 0.15})

# Move the files into the appropriate directories
move_files(labels_path, labels_path / 'train', train_files, images_path)
move_files(labels_path, labels_path / 'test', test_files, images_path)
move_files(labels_path, labels_path / 'val', val_files, images_path)

