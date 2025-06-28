import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("../model/mask_detector_model.h5")

# Define class labels
labels = ["Mask", "No Mask"]

# Load your test image (place the image in the src/ folder or give full path)
img = cv2.imread("test.jpg")  # 👈 change this to your image filename
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (100, 100))  # Use 128 if you trained on 128x128
img = img / 255.0
img = np.reshape(img, (1, 100, 100, 3))  # or (1, 128, 128, 3) if changed

# Make prediction
prediction = model.predict(img)
label = np.argmax(prediction)
confidence = prediction[0][label]

print(f"Prediction: {labels[label]} ({confidence:.2f} confidence)")
