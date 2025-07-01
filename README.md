# Face Mask Detection 🚀

## 📌 Objective
To build a deep learning system that detects whether a person is wearing a face mask or not in real-time using a webcam and also through image uploads via a web interface.

---

## 🛠️ Tools & Libraries Used
- Python 3  
- TensorFlow / Keras – for CNN model  
- OpenCV – for real-time webcam detection  
- Flask – for web interface  
- NumPy, Matplotlib – for preprocessing and visualization  
- scikit-learn – for label encoding and preprocessing

---

## 📁 Project Structure
face_mask_detectionn/
│
├── dataset/ # (optional) raw training data
├── images/ # test images for predictions
├── model/
│ └── mask_detector_model.keras # trained CNN model
│
├── src/
│ ├── cnn_model.py # CNN model architecture
│ ├── data_preprocess.py # preprocessing logic
│ └── detect_mask_webcam.py # real-time webcam detection script
│
├── webapp/
│ ├── static/
│ │ └── uploaded/ # uploaded test images
│ │ └── output_sample.png # 📸 Web output sample image
│ ├── templates/
│ │ └── index.html # Flask web UI
│ ├── app.py # Flask server
│
├── live_detect.py # alternative webcam live detection
├── README.md # ✅ Project documentation


---

## ⚙️ Project Workflow (Code-wise)

### ✅ Step 1: Model Training
- CNN model is trained using labeled images (`With Mask`, `Without Mask`)  
- Data is resized (128x128), normalized, and passed through Conv2D layers  
- Final model is saved as `.keras` in `model/`

### ✅ Step 2: Web App for Mask Detection
- Flask app (`app.py`) serves a simple UI for uploading images  
- Uploaded image is saved, preprocessed, and passed to the trained model  
- Prediction is displayed with confidence score

### ✅ Step 3: Real-Time Webcam Detection
- `detect_mask_webcam.py` opens the webcam  
- Face is detected using OpenCV’s `haarcascade_frontalface_default.xml`  
- Each face region is passed to the model for mask detection  
- Output window shows bounding boxes with labels: “Mask” or “No Mask”

---

## 🌐 Web App Output Sample

![Screenshot 2025-06-29 002943](https://github.com/user-attachments/assets/17501f3f-e1c1-400c-b3dd-0c1129515abe)

---

## 🖥️ How to Run the Project Locally

### 🧾 Clone the Repository

```bash
git clone https://github.com/mamidalasathvika15/face_mask_detectionn.git
cd face_mask_detectionn

📦 Create Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

📥 Install Requirements

pip install -r requirements.txt

🚀 Run the Flask Web App

cd webapp
python app.py
Then open your browser and go to http://127.0.0.1:5000

🎥 Run the Live Webcam Detection
cd src
python detect_mask_webcam.py

📊 Output & Results
✅ CNN model achieves high accuracy on training data
✅ Real-time mask detection with webcam
✅ User-friendly web interface with prediction results
✅ Bounding boxes drawn around faces with labels

📌 Conclusion
This project successfully detects face masks in real-time and through user-uploaded images using a deep learning model and OpenCV. It provides both a GUI (web) and live detection experience. Can be enhanced further with dataset expansion or transfer learning.

 M. Sathvika
 Date: 28-06-2025

