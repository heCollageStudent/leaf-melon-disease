from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import cv2
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.feature import graycomatrix, graycoprops

# ==== Inisialisasi Flask ====
app = Flask(__name__)

# ==== Load model dan tools ====
model = load_model('model_special.h5', compile=False)
cnn_base = load_model('mobilenetv2_imagenet.keras', compile=False)
scaler_warna = joblib.load('scaler_warna.pkl')
scaler_tekstur = joblib.load('scaler_tekstur.pkl')
scaler_bentuk = joblib.load('scaler_bentuk.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# ==== Fungsi ekstraksi fitur manual ====
def extract_fitur_manual(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    if np.sum(mask) == 0:
        raise ValueError("Tidak ada piksel daun terdeteksi.")

    L = img_lab[..., 0].astype(np.float32)[mask]
    a = img_lab[..., 1].astype(np.float32)[mask]
    b = img_lab[..., 2].astype(np.float32)[mask]
    fitur_warna = np.array([
        np.mean(L), np.std(L),
        np.mean(a), np.std(a),
        np.mean(b), np.std(b)
    ])

    gray_rs = cv2.resize(gray, (128,128))
    glcm = graycomatrix(gray_rs, distances=[1,2,3,4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    fitur_tekstur = np.array([
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean()
    ])

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("Kontur daun tidak ditemukan.")
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0
    circularity = 4 * np.pi * area / (perimeter**2) if perimeter != 0 else 0
    rectangularity = area / (w * h) if w*h != 0 else 0
    diameter = np.sqrt((4 * area) / np.pi)
    fitur_bentuk = np.array([area, perimeter, w, h, aspect_ratio, circularity, rectangularity, diameter])

    return fitur_warna, fitur_tekstur, fitur_bentuk

# ==== Endpoint prediksi ====
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file'].read()
    try:
        # Decode image
        img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Image could not be read"}), 400

        # CNN
        cnn_img = cv2.resize(img, (224, 224))
        cnn_img = img_to_array(cnn_img)
        cnn_img = preprocess_input(cnn_img)
        cnn_img = np.expand_dims(cnn_img, axis=0)
        fitur_cnn = cnn_base.predict(cnn_img, verbose=0)[0]

        # Manual
        warna, tekstur, bentuk = extract_fitur_manual(img)
        warna_scaled   = scaler_warna.transform([warna])[0]
        tekstur_scaled = scaler_tekstur.transform([tekstur])[0]
        bentuk_scaled  = scaler_bentuk.transform([bentuk])[0]
        fitur_manual = np.concatenate([warna_scaled, tekstur_scaled, bentuk_scaled])

        # Gabung & prediksi
        combined_input = [np.expand_dims(fitur_cnn, axis=0), np.expand_dims(fitur_manual, axis=0)]
        probs = model.predict(combined_input, verbose=0)[0]
        pred_index = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        prob_dict = {label_encoder.inverse_transform([i])[0]: round(float(p) * 100, 2) for i, p in enumerate(probs)}

        return jsonify({
            "prediction": pred_label,
            "probabilities": prob_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== Root endpoint ====
@app.route("/")
def home():
    return "✅ Hybrid CNN + Manual Feature Leaf Classifier API"

# ==== Run ====
if __name__ == "__main__":
    app.run(debug=True)
