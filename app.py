from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)

# Muat model sekali saja ketika aplikasi dijalankan
model_path = os.getenv('MODEL_PATH', 'model/face-shape-recognizer.h5')
model = load_model(model_path)

def predict_label(img):
    img_ori = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    img_rsz = cv2.resize(img_ori, (190, 250))
    img_gray = cv2.cvtColor(img_rsz, cv2.COLOR_BGR2GRAY)
    i = img_gray / 255.0
    i = np.expand_dims(i, axis=0)

    labels = {
        0: 'Heart',
        1: 'Oblong',
        2: 'Oval',
        3: 'Round',
        4: 'Square'
    }

    predictions = np.argmax(model.predict(i))

    return labels[predictions], img_ori

def encode_img(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return img_str

@app.route("/predict", methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file format'}), 400

    img = file.read()
    
    try:
        prediction, img_ori = predict_label(img)
        encoded_image = encode_img(img_ori)
        return jsonify({'prediction': prediction, 'image': encoded_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
