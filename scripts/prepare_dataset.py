"""Prepare dataset: organize images into breed folders and generate augmented copies."""
import os, shutil, random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_dataset')

BREED_MAPPING = {
    'gir-cow-1.jpg': 'Gir',
    'Jaffarabadi_1.jpg': 'Jaffarabadi',
    'Murrah-Buffalo.jpg': 'Murrah',
    'red_sindhi_bull_1.jpg': 'Red_Sindhi',
    'Sahiwal_1.jpg': 'Sahiwal',
}

def augment_image(image):
    """Return list of (name, augmented_image) tuples."""
    augs = [('original', image.copy()), ('hflip', ImageOps.mirror(image))]
    for angle in [15, 30, 45, -15, -30]:
        augs.append((f'rot_{angle}', image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(128,128,128))))
    for factor in [0.7, 0.85, 1.15, 1.3]:
        augs.append((f'bright_{factor}', ImageEnhance.Brightness(image).enhance(factor)))
    for factor in [0.8, 1.2]:
        augs.append((f'contrast_{factor}', ImageEnhance.Contrast(image).enhance(factor)))
        augs.append((f'sat_{factor}', ImageEnhance.Color(image).enhance(factor)))
    return augs

def main():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR)

    for image_file, breed in BREED_MAPPING.items():
        src = os.path.join(DATASET_DIR, image_file)
        if not os.path.exists(src):
            print(f"Warning: {src} not found")
            continue
        print(f"\nProcessing: {image_file} -> {breed}")
        with Image.open(src) as img:
            img = img.convert('RGB').resize((224, 224), Image.LANCZOS)
            samples = augment_image(img)
            # Add extra random variations
            for i in range(30):
                aug = img.copy()
                if np.random.random() > 0.3:
                    aug = aug.rotate(np.random.uniform(-40, 40), resample=Image.BICUBIC, expand=True, fillcolor=(128,128,128))
                if np.random.random() > 0.5:
                    aug = ImageOps.mirror(aug)
                if np.random.random() > 0.3:
                    aug = ImageEnhance.Brightness(aug).enhance(np.random.uniform(0.6, 1.4))
                if np.random.random() > 0.3:
                    aug = ImageEnhance.Contrast(aug).enhance(np.random.uniform(0.7, 1.3))
                if np.random.random() > 0.3:
                    aug = ImageEnhance.Color(aug).enhance(np.random.uniform(0.6, 1.4))
                samples.append((f'custom_{i}', aug.copy()))

            random.shuffle(samples)
            num_val = max(1, int(len(samples) * 0.2))
            val_samples, train_samples = samples[:num_val], samples[num_val:]

            train_dir = os.path.join(PROCESSED_DIR, 'train', breed)
            val_dir = os.path.join(PROCESSED_DIR, 'val', breed)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            base = image_file.replace('.jpg', '')
            for idx, (name, aug_img) in enumerate(train_samples):
                aug_img.resize((224, 224), Image.LANCZOS).save(os.path.join(train_dir, f'{base}_{name}_{idx}.jpg'), 'JPEG', quality=95)
            for idx, (name, aug_img) in enumerate(val_samples):
                aug_img.resize((224, 224), Image.LANCZOS).save(os.path.join(val_dir, f'{base}_{name}_{idx}.jpg'), 'JPEG', quality=95)
            print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    print("\nDataset preparation complete!")
    total_train = total_val = 0
    for split in ['train', 'val']:
        split_dir = os.path.join(PROCESSED_DIR, split)
        for breed in sorted(os.listdir(split_dir)):
            count = len([f for f in os.listdir(os.path.join(split_dir, breed)) if f.endswith('.jpg')])
            if split == 'train': total_train += count
            else: total_val += count
            print(f"  {split}/{breed}: {count}")
    print(f"\nTotal train: {total_train}, Total val: {total_val}, Total: {total_train + total_val}")

if __name__ == '__main__':
    main()