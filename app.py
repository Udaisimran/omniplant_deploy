from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import io
import gdown

app = Flask(_name_)

# Path to save the TFLite model
model_path = "model.tflite"

# If the model is not already downloaded, download it from Google Drive
if not os.path.exists(model_path):
    file_id = "1o4y1x9UG4ZS8D0fstDsfOsG5yUFSNRb1"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# HARDCODED CLASS NAMES (NOT RECOMMENDED)
class_labels = [
    "Apple__Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple__healthy",
    "Blueberry__healthy", "Cherry(including_sour)Powdery_mildew", "Cherry(including_sour)_healthy",
    "Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot", "Corn(maize)Common_rust",
    "Corn_(maize)Northern_Leaf_Blight", "Corn(maize)healthy", "Grape__Black_rot",
    "Grape__Esca(Black_Measles)", "Grape__Leaf_blight(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange__Haunglongbing(Citrus_greening)", "Peach__Bacterial_spot", "Peach__healthy",
    "Pepper,bell_Bacterial_spot", "Pepper,_bell_healthy", "Potato_Early_blight", "Potato__Late_blight",
    "Potato__healthy", "Raspberry_healthy", "Soybean_healthy", "Squash__Powdery_mildew",
    "Strawberry__Leaf_scorch", "Strawberry_healthy", "Tomato_Bacterial_spot", "Tomato__Early_blight",
    "Tomato__Late_blight", "Tomato_Leaf_Mold", "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites Two-spotted_spider_mite", "Tomato_Target_Spot", "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus", "Tomato__healthy"
]
  # Replace with your actual class names!

@app.route("/")
def home():
    return "Flask Model Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # Preprocess image
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Set the tensor to the input tensor of the model
        input_data = np.array(img_array, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the prediction results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0].tolist()

        # Get the class with the highest prediction score
        predicted_class_index = prediction.index(max(prediction))
        predicted_class_label = class_labels[predicted_class_index]

        return jsonify({"prediction": prediction, "predicted_class": predicted_class_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
