

import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from pymongo import MongoClient
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# For MobileNetV2-based leaf detection
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)

# Set upload folder (inside static/uploads)
UPLOAD_FOLDER = os.path.join("static", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# MongoDB Connection
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["plantdisease"]
diagnosis_collection = db["diagnosis"]
cures_collection = db["cures"]
commondisease_collection = db["commondisease"]

# Load the prediction model
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "plant_disease_prediction_model.h5")
disease_model = tf.keras.models.load_model(model_path)

# List of disease classes (must match your model's output)
classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype("float32") / 255.0

def predict_disease(image_path):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = disease_model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_disease = classes[predicted_index]
    diagnosis_doc = diagnosis_collection.find_one({"disease": predicted_disease})
    diagnosis_desc = diagnosis_doc.get("description", "No diagnosis description available.") if diagnosis_doc else "No diagnosis description available."
    sample_images = diagnosis_doc.get("image_paths", []) if diagnosis_doc else []
    return predicted_disease, diagnosis_desc, sample_images

# Optional Leaf Detection Functions
leaf_classifier = MobileNetV2(weights="imagenet")
plant_names = {cls.split("___")[0].lower() for cls in classes}
generic_keywords = {"leaf", "plant", "tree", "flower", "daisy", "rose", "sunflower", "tulip", "lily", "fern", "ivy", "moss", "orchid", "palm", "carnation", "dandelion", "bougainvillea", "bush", "shrub", "vine", "grapevine", "crop", "vegetable"}
allowed_keywords = list(plant_names.union(generic_keywords))

def detect_leaf_by_imagenet(image_path, top=10, prob_threshold=0.05):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = leaf_classifier.predict(x)
    decoded = decode_predictions(preds, top=top)[0]
    max_allowed_prob = 0.0
    for _, label, prob in decoded:
        for keyword in allowed_keywords:
            if keyword in label.lower():
                max_allowed_prob = max(max_allowed_prob, prob)
    return max_allowed_prob >= prob_threshold

def detect_leaf_by_color(image_path, area_threshold=0.05):
    img = cv2.imread(image_path)
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    max_area = max(cv2.contourArea(c) for c in contours)
    total_area = img.shape[0] * img.shape[1]
    return (max_area / total_area) >= area_threshold

def is_leaf_image(image_path):
    return detect_leaf_by_imagenet(image_path, top=10, prob_threshold=0.05) or detect_leaf_by_color(image_path, area_threshold=0.05)

# Custom Route to Serve Files from "plant dataset" (folder name with a space)
@app.route("/plantdataset/<path:filename>")
def plant_dataset(filename):
    folder_path = os.path.join(app.root_path, "plant dataset")
    return send_from_directory(folder_path, filename)

# Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return redirect(request.url)
        file = request.files["file"]
        filename = file.filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        if not is_leaf_image(file_path):
            error_message = "Invalid Image: No plant leaf detected. Please upload a clear image of a plant leaf."
            return render_template("prediction.html", error=error_message)
        predicted_disease, diagnosis_desc, sample_images = predict_disease(file_path)
        return render_template("prediction.html",
                               disease=predicted_disease,
                               diagnosis=diagnosis_desc,
                               sample_images=sample_images,
                               file_path=filename)
    return render_template("prediction.html")

@app.route("/cure")
def cure():
    disease = request.args.get("disease", None)
    if disease:
        cure_doc = cures_collection.find_one({"disease": disease})
        if cure_doc:
            biological = cure_doc.get("biological", "No biological solution available.")
            chemical = cure_doc.get("chemical", "No chemical solution available.")
        else:
            biological = chemical = "No cure information available for this disease."
    else:
        biological = chemical = "No disease selected."
    return render_template("cure.html", disease=disease, biological=biological, chemical=chemical)

@app.route("/gettoknow", methods=["GET", "POST"])
def gettoknow():
    common_diseases = []
    selected_plant = None
    if request.method == "POST":
        selected_plant = request.form.get("plant")
        doc = commondisease_collection.find_one({"plant": selected_plant})
        if doc:
            common_diseases = doc.get("diseases", [])
    plant_names = commondisease_collection.distinct("plant")
    return render_template("gettoknow.html", plants=plant_names, selected=selected_plant, diseases=common_diseases)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)


