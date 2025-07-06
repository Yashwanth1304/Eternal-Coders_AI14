# GreenScan – AI-Based Crop Disease Detection and Cure Assistant  
**Presented at INFOTHON 4.0**

**GreenScan** is an intelligent web-based system designed to help farmers and agricultural workers identify crop diseases early by simply uploading a leaf image. It provides accurate disease detection, relevant cure suggestions, and even chatbot support for unknown cases — all without needing expert intervention.

---

## What GreenScan Offers

- **Leaf Image Diagnosis** – Upload images of infected plant leaves to get disease predictions.  
- **AI-Based Detection** – Uses a deep learning model to identify 38 different crop diseases.  
- **Cure Recommendations** – Displays relevant cures and treatments stored in the database.  
- **Chatbot Support** – If the disease is unknown, describe the symptoms to get NLP-based help.  
- **Farmer Friendly** – Provides guidance in simple language, minimizing dependence on agronomists.  

---

## How It Helps

### GreenScan empowers farmers to:

- Detect diseases early, reducing crop loss.  
- Get reliable, instant solutions anytime, anywhere.  
- Make informed decisions without waiting for expert visits.  
- Use an easy web interface built for both mobile and desktop.

---

## Technologies Used

| **Category**         | **Tools/Frameworks Used**                      |
|----------------------|------------------------------------------------|
| Deep Learning         | TensorFlow, MobileNetV2                        |
| Image Processing      | OpenCV, Pillow, NumPy                          |
| Backend/API           | Flask (Python)                                 |
| Database              | MongoDB, PyMongo                               |
| Chatbot/NLP           | Custom NLP logic for user-described symptoms   |
| Frontend              | HTML, CSS (via Flask templates)                |
| Development Tools     | Jupyter Notebook, Postman, dotenv              |

---
## Technical Workflow

GreenScan follows a well-structured AI pipeline to detect crop diseases from images and provide appropriate cure recommendations.

### 🔁 How It Works

1. **Image Upload via Web Interface**  
   Users upload a plant leaf image through the Flask-based web UI.

2. **Image Preprocessing**  
   The uploaded image is resized, normalized, and converted to array format using OpenCV, Pillow, and NumPy.

3. **AI-Based Prediction**  
   The pre-trained MobileNetV2 (TensorFlow) model predicts the disease from 38 trained classes with high accuracy.

4. **Cure Retrieval from MongoDB**  
   Based on the predicted disease, the system queries MongoDB to fetch pre-stored cure details and displays them.

5. **Fallback: Chatbot for Unrecognized Symptoms**  
   If the disease isn’t confidently predicted, users can describe the symptoms in text. The integrated NLP chatbot parses the description and maps it to known diseases using a keyword-based approach, then suggests matching cures.

---

### 🗂️ Workflow Summary
User Uploads Image
        ↓
Flask Receives Image
        ↓
Image Preprocessed (Pillow, OpenCV)
        ↓
MobileNetV2 Model Predicts Disease
        ↓
MongoDB Returns Cure for Prediction
        ↓
Cure + Info Displayed to User
        ↓
(If unknown → NLP Chatbot → Symptom Match → Cure)

