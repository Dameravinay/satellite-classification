import os
import gdown
from flask import Flask, render_template, request
import numpy as np
import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# ===============================
# MODEL DOWNLOAD (from Google Drive)
# ===============================
MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=1gX0aC9-eMoxNZA0uiBPpn_YhG7op5-zb"
    gdown.download(url, MODEL_PATH, quiet=False)

# ===============================
# LOAD MODEL (only once)
# ===============================
model = keras.models.load_model(MODEL_PATH)

# ===============================
# CLASS LABELS
# ===============================
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
           'River', 'SeaLake']


# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return predicted_class, confidence


# ===============================
# ROUTES
# ===============================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']

        if file:
            image_path = os.path.join("static", file.filename)
            file.save(image_path)

            prediction, confidence = predict_image(image_path)

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path)


# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)