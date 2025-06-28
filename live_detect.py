import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/mask_detector_model.keras")

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))  # Match model input size
        face_normalized = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        prediction = model.predict(face_input)[0][0]  # value between 0 and 1

        # Adjust threshold to reduce false positives
        threshold = 0.90

        if prediction >= threshold:
            label = "Mask"
            confidence = prediction
            color = (0, 255, 0)  # Green
        else:
            label = "No Mask"
            confidence = 1 - prediction
            color = (0, 0, 255)  # Red

        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Live Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
