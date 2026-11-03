# Cattle & Buffalo Breed Recognition System

An AI-powered image recognition system that identifies cattle and buffalo breeds from images using deep learning (ResNet-18 transfer learning).

## Supported Breeds

| Breed | Species | Origin |
|-------|---------|--------|
| Gir | Cattle (Bos indicus) | Gujarat, India |
| Jaffarabadi | Buffalo (Bubalus bubalis) | Gujarat, India |
| Murrah | Buffalo (Bubalus bubalis) | Haryana & Punjab, India |
| Red Sindhi | Cattle (Bos indicus) | Sindh, Pakistan / Rajasthan, India |
| Sahiwal | Cattle (Bos indicus) | Punjab, India & Pakistan |

## Project Structure

```
sih_project/
├── app.py                    # Flask web application
├── templates/
│   └── index.html            # Web UI
├── scripts/
│   ├── prepare_dataset.py    # Dataset preparation & augmentation
│   ├── train_model.py        # Model training script
│   ├── predict.py            # Prediction/inference script
│   └── inspect_dataset.py    # Dataset inspection utility
├── dataset/                  # Original dataset images
├── processed_dataset/        # Augmented organized dataset (auto-generated)
├── models/                   # Trained model files (auto-generated)
└── requirements.txt          # Python dependencies
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torchvision 0.15+
- Flask 3.0+
- Pillow, NumPy, scikit-learn, matplotlib, tqdm

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For CPU-only PyTorch on Windows (if standard install fails):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Usage

### 1. Prepare the Dataset
```bash
python scripts/prepare_dataset.py
```
This organizes images into breed folders and generates augmented training samples.

### 2. Train the Model
```bash
python scripts/train_model.py
```
Trains a ResNet-18 model using transfer learning. The best model is saved to `models/breed_model.pth` along with training history and curves.

### 3. Make Predictions via CLI
```bash
python scripts/predict.py path/to/image.jpg
```

This outputs the predicted breed, confidence score, probability distribution, and detailed breed information.

### 4. Run Web Application
```bash
python app.py
```
Then open your browser at `http://127.0.0.1:5000`

Upload an image of a cattle or buffalo to automatically identify its breed with confidence scores and detailed breed information.

## How It Works

1. **Data Augmentation**: Original images are augmented with rotations, flips, brightness/contrast/saturation adjustments to create a larger training dataset (225 total samples).
2. **Transfer Learning**: Uses ResNet-18 pre-trained on ImageNet. All convolutional layers are frozen, and a new classifier head is trained on the breed dataset.
3. **Prediction**: The model processes the image through the same preprocessing pipeline and outputs probability scores for each breed class.

## Model Performance

The model achieves:
- ~99% training accuracy
- 100% validation accuracy
- Real-time prediction speed on CPU

## Notes

- The dataset currently contains 5 breeds with 1 image each, augmented to 45 samples per breed
- For production use, more images per breed should be collected
- The model is trained on the available dataset and can be extended to more breeds
# breed_recognition
