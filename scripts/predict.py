"""Predict the breed of cattle/buffalo from an image using the trained model."""
import os, sys, json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'breed_model.pth')

# Breed information database
BREED_INFO = {
    'Gir': {
        'species': 'Cattle (Bos indicus)',
        'origin': 'Gujarat, India (Gir forest region)',
        'characteristics': 'Distinctive arched forehead, long drooping ears, white to dark red coat with white patches',
        'milk_production': 'High milk yield (up to 40+ litres/day in peak lactation)',
        'note': 'One of India\'s best dairy breeds, known for hardiness and disease resistance'
    },
    'Jaffarabadi': {
        'species': 'Buffalo (Bubalus bubalis)',
        'origin': 'Gujarat, India (Saurashtra region)',
        'characteristics': 'Massive body, long horns that sweep backwards, black skin with sparse hair',
        'milk_production': 'Very high milk yield (18-20 litres/day), rich butterfat content',
        'note': 'Largest buffalo breed in India, excellent draught animal'
    },
    'Murrah': {
        'species': 'Buffalo (Bubalus bubalis)',
        'origin': 'Haryana & Punjab, India',
        'characteristics': 'Jet black coat, tightly curved horns, wedge-shaped body',
        'milk_production': 'Excellent milk yield (10-16 litres/day), high fat (6-7%)',
        'note': 'India\'s most popular dairy buffalo breed, widely exported'
    },
    'Red_Sindhi': {
        'species': 'Cattle (Bos indicus)',
        'origin': 'Sindh province, Pakistan (now in Rajasthan, India)',
        'characteristics': 'Reddish-brown coat, lyre-shaped horns, loose dewlap',
        'milk_production': 'Good milk yield (10-15 litres/day), excellent heat tolerance',
        'note': 'Prized for heat tolerance, disease resistance, and quality milk'
    },
    'Sahiwal': {
        'species': 'Cattle (Bos indicus)',
        'origin': 'Punjab region (India & Pakistan)',
        'characteristics': 'Reddish-dun coat, short horns, large hump, heavy dewlap',
        'milk_production': 'Excellent milk yield (15-20 litres/day), among best Indian dairy cattle',
        'note': 'The best dairy cattle breed in India and Pakistan'
    }
}


def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Please run scripts/train_model.py first.")
        sys.exit(1)
    
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    class_names = checkpoint['class_names']
    
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(model.fc.in_features, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, len(class_names)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names


def predict_image(model, class_names, image_path, return_probs=False):
    """Predict the breed from an image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probabilities, 0)
    
    breed = class_names[pred_idx.item()]
    confidence = conf.item()
    probs = {class_names[i]: probabilities[i].item() for i in range(len(class_names))}
    
    if return_probs:
        return breed, confidence, probs
    return breed, confidence


def get_breed_info(breed):
    """Get the information for a breed."""
    if breed == 'Red_Sindhi':
        display_name = 'Red Sindhi'
    else:
        display_name = breed
    info = BREED_INFO.get(breed, {})
    if not info:
        info = {
            'species': 'Unknown',
            'origin': 'Unknown',
            'characteristics': 'No data available',
            'milk_production': 'Unknown',
            'note': 'No additional information'
        }
    return display_name, info


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    print(f"Loading model...")
    model, class_names = load_model()
    
    breed, confidence, probs = predict_image(model, class_names, image_path, return_probs=True)
    display_name, info = get_breed_info(breed)
    
    print(f"\n{'='*60}")
    print(f"Prediction Results")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"\nPredicted Breed: {display_name}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"\n--- All probabilities ---")
    for name, prob in sorted(probs.items(), key=lambda x: -x[1]):
        display = 'Red Sindhi' if name == 'Red_Sindhi' else name
        print(f"  {display}: {prob*100:.2f}%")
    
    print(f"\n--- Breed Details ---")
    print(f"Species: {info['species']}")
    print(f"Origin: {info['origin']}")
    print(f"Characteristics: {info['characteristics']}")
    print(f"Milk Production: {info['milk_production']}")
    print(f"Note: {info['note']}")


if __name__ == '__main__':
    main()