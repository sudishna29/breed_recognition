"""Inspect the dataset images and report their properties."""
import os
from PIL import Image

dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')
dataset_dir = os.path.abspath(dataset_dir)

print("Dataset directory:", dataset_dir)
print("=" * 60)

image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

if not image_files:
    print("No images found in dataset directory!")
else:
    print(f"Found {len(image_files)} images:")
    print("=" * 60)
    for img_file in sorted(image_files):
        img_path = os.path.join(dataset_dir, img_file)
        try:
            with Image.open(img_path) as img:
                print(f"  {img_file}")
                print(f"    Size: {img.size[0]}x{img.size[1]}")
                print(f"    Mode: {img.mode}")
                print(f"    Format: {img.format}")
                print()
        except Exception as e:
            print(f"  {img_file} - ERROR: {e}")
            print()