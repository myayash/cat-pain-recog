from flask import Flask, request, jsonify
from preprocessing import preprocess_img
from model_setup import build_detect_cat, detect_cats, build_model, predict
import torch
from werkzeug.utils import secure_filename 
import os


app = Flask(__name__)

print('CHECKING DEVICE...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('GETTING MODEL FILE PATH...')
model_path = './cat-pr-rn18-adamW-FL.pth'

print ('GETTING DETECTOR MODEL FILE PATH...')
detection_model_path = './detector'

thr = 0.4

detect_model =  build_detect_cat(detection_model_path, verbose=True)
model = build_model(model_path, device=device, verbose=True)

@app.route('/')
def home():
    return 'API is running... zzZzzaooWw!'


@app.route('/predict', methods=['POST'])
def predict_pain():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify ({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join('/tmp', filename)
    file.save(temp_path)

    image_tensor = preprocess_img(temp_path, device=device, verbose=False)

    os.remove(temp_path)

    detection_result = detect_cats(image_tensor, detect_model, verbose=False)

    if detection_result == 'CAT':
        prediction = predict(image_tensor, model, thr=thr, verbose=False)
        return jsonify({'image': 'CAT', 'prediction': prediction})
    else:
        return jsonify({'image': 'NOT_CAT', 'message': 'Get a cat image plzz.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
