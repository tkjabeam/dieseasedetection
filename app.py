from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load your pre-trained model
model_path = 'skin_diseases.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Define the class labels in the correct order
class_labels = [
    'Atopic Dermatitis', 
    'Eczema', 
    'Melanocytic Nevi', 
    'Psoriasis pictures Lichen Planus and related diseases', 
    'Seborrheic Keratoses and other Benign Tumors', 
    'Tinea Ringworm Candidiasis and other Fungal Infections', 
    'Warts Molluscum and other Viral Infections'
]

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = 'uploaded_image.jpg'
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions[0])]

        return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
