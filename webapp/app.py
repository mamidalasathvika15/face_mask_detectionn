from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploaded"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model with 128x128 input shape
model = load_model("../model/mask_detector_model.keras")

IMG_SIZE = 128
labels = ["Mask", "No Mask"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            result = "❌ Failed to read uploaded image."
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

            prediction = model.predict(img)
            label = int(np.round(prediction[0][0]))
            confidence = prediction[0][0] if label == 1 else 1 - prediction[0][0]
            result = f"{labels[label]} ({confidence:.2f} confidence)"
            image_path = filepath

    return render_template("index.html", result=result, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
