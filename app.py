"""Web application for cattle and buffalo breed recognition."""
import os
import sys
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from predict import load_model, predict_image, get_breed_info, BREED_INFO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load model at startup
MODEL = None
CLASS_NAMES = None

def init_model():
    global MODEL, CLASS_NAMES
    if MODEL is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'breed_model.pth')
        if not os.path.exists(model_path):
            print("WARNING: Model not found. Please train the model first with scripts/train_model.py")
            return
        MODEL, CLASS_NAMES = load_model()
        print(f"Model loaded. Classes: {CLASS_NAMES}")

init_model()


@app.route('/')
def index():
    breeds = []
    for key, info in BREED_INFO.items():
        display = 'Red Sindhi' if key == 'Red_Sindhi' else key
        breeds.append({
            'name': display,
            'species': info['species'],
            'origin': info['origin']
        })
    return render_template('index.html', breeds=breeds, model_ready=MODEL is not None)


@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Read and validate image
        image = Image.open(file.stream).convert('RGB')
        
        # Save to temp file and predict
        temp_path = os.path.join(app.root_path, 'temp_upload.jpg')
        image.save(temp_path, 'JPEG')

        breed, confidence, probs = predict_image(MODEL, CLASS_NAMES, temp_path, return_probs=True)
        display_name, info = get_breed_info(breed)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Convert probs dict for JSON
        probs_list = []
        for name, prob in sorted(probs.items(), key=lambda x: -x[1]):
            display = 'Red Sindhi' if name == 'Red_Sindhi' else name
            probs_list.append({'breed': display, 'probability': round(prob * 100, 2)})

        return jsonify({
            'success': True,
            'breed': display_name,
            'confidence': round(confidence * 100, 2),
            'species': info['species'],
            'origin': info['origin'],
            'characteristics': info['characteristics'],
            'milk_production': info['milk_production'],
            'note': info['note'],
            'probabilities': probs_list
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)