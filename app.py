from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import cv2
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ==== Load model dan tools ====
model = load_model('model_special.h5')
scaler_warna = joblib.load('saved_scalers/scaler_warna.pkl')
scaler_tekstur = joblib.load('saved_scalers/scaler_tekstur.pkl')
scaler_bentuk = joblib.load('saved_scalers/scaler_bentuk.pkl')
label_encoder = joblib.load('saved_scalers/label_encoder.pkl')
cnn_base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))

def extract_features(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    # --- CNN ---
    cnn_img = cv2.resize(img, (224, 224))
    cnn_img = img_to_array(cnn_img)
    cnn_img = preprocess_input(cnn_img)
    cnn_img = np.expand_dims(cnn_img, axis=0)
    fitur_cnn = cnn_base.predict(cnn_img)[0]

    # --- Manual (warna, tekstur, bentuk) ---
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    L = img_lab[..., 0][mask]
    a = img_lab[..., 1][mask]
    b = img_lab[..., 2][mask]
    warna = np.array([np.mean(L), np.std(L), np.mean(a), np.std(a), np.mean(b), np.std(b)])
    gray_rs = cv2.resize(gray, (128,128))
    glcm = graycomatrix(gray_rs, distances=[1,2,3,4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    tekstur = np.array([
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean()
    ])
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0
    circularity = 4 * np.pi * area / (perimeter**2) if perimeter != 0 else 0
    rectangularity = area / (w*h) if w*h != 0 else 0
    diameter = np.sqrt((4 * area) / np.pi)
    bentuk = np.array([area, perimeter, w, h, aspect_ratio, circularity, rectangularity, diameter])

    # Scaling
    warna_scaled = scaler_warna.transform([warna])[0]
    tekstur_scaled = scaler_tekstur.transform([tekstur])[0]
    bentuk_scaled = scaler_bentuk.transform([bentuk])[0]
    fitur_manual = np.concatenate([warna_scaled, tekstur_scaled, bentuk_scaled])

    return fitur_cnn, fitur_manual

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file'].read()
    try:
        cnn, manual = extract_features(file)
        result = model.predict([np.expand_dims(cnn, axis=0), np.expand_dims(manual, axis=0)])[0]
        probs = {label_encoder.inverse_transform([i])[0]: round(float(prob)*100, 2) for i, prob in enumerate(result)}
        label = max(probs, key=probs.get)
        return jsonify({"prediction": label, "probabilities": probs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Hybrid CNN + Manual Feature API"

if __name__ == "__main__":
    app.run(debug=True)
