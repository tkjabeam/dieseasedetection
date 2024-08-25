from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load your pre-trained model
model_path = 'skin_diseases.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)
logging.info(f"Model loaded from {model_path}")

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
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
        # Ensure the uploads directory exists
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the uploaded file to the uploads directory
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        logging.info(f"File saved to {file_path}")

        try:
            # Preprocess the image
            img_array = preprocess_image(file_path)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions[0])]
            logging.info(f"Prediction made: {predicted_class}")

            return jsonify({'prediction': predicted_class})

        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
