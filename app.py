import os
import gdown
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io

# Flask app
app = Flask(__name__)

# Google Drive TFLite model URL
GDRIVE_URL = "https://drive.google.com/uc?id=1o4y1x9UG4ZS8D0fstDsfOsG5yUFSNRb1"
MODEL_PATH = "efficient_model.tflite"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load TFLite model and create interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Define input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Define class labels (ensure this matches your model's output order)
class_labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]
# Route for testing if the app is running
@app.route('/')
def home():
    return 'Welcome to the Plant Disease Prediction API!'
# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)

        prediction = model.predict(image)
        predicted_index = np.argmax(prediction)
        predicted_label = class_labels[predicted_index]

        result = {
            'predicted_class': predicted_label,
            'confidence_scores': prediction[0].tolist()
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app on port 5000
if __name__ == '__main__':
    app.run(debug=True, port=5000)
