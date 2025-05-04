import os

import torch
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from ldm_detect import process_response, send_img
from model_setup import build_detect_cat, build_model, detect_cats, predict
from preprocess import preprocess_img

app = Flask(__name__)

print("CHECKING DEVICE...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("GETTING MODEL FILE PATH...")
MODEL_PATH = "./cat-pr-rn18-adamW-FL.pth"

print("GETTING DETECTOR MODEL FILE PATH...")
DETECTION_MODEL_PATH = "./detector"

print("SETTING LANDMARK API...")
API_URL = "http://34.165.76.57:6000/landmarks"
print("SETTING THRESHOLD...")
THR = 0.4

detect_model = build_detect_cat(DETECTION_MODEL_PATH)
model = build_model(MODEL_PATH, device=device)


@app.route("/")
def home():
    return "API is running... zzZzzaooWw!"


@app.route("/predict", methods=["POST"])
def predict_pain():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join("/tmp", filename)
    file.save(temp_path)

    image_tensor = preprocess_img(temp_path, device=device, plot=True, verbose=True)
    detection_result = detect_cats(image_tensor, detect_model)

    if detection_result == "CAT":
        # scaled image cant get ldms detected properly
        result = send_img(temp_path, API_URL, verbose=True)
        cropped_image = process_response(
            result, temp_path, verbose=True, conv_tensor=True
        )
        os.remove(temp_path)

        prediction = predict(cropped_image, model, thr=THR, verbose=True)

        return jsonify({"image": "CAT", "prediction": prediction})
    else:
        return jsonify({"image": "NOT_CAT", "message": "Get a cat image plzz."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
