import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
print("🔁 Loading model...")
model = load_model("../model/mask_detector_model.keras")
print("✅ Model loaded!")

# Set up labels and image size
labels = ["Mask", "No Mask"]
IMG_SIZE = 100

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📸 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
        face_normalized = face_resized / 255.0
        face_input = np.reshape(face_normalized, (1, IMG_SIZE, IMG_SIZE, 3))

        prediction = model.predict(face_input)[0][0]
        label_index = int(np.round(prediction))
        confidence = prediction if label_index == 1 else 1 - prediction
        label_text = f"{labels[label_index]} ({confidence:.2f})"

        color = (0, 255, 0) if label_index == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Face Mask Detection - Press 'q' to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exited.")
