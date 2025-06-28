import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Paths
DATASET_PATH = "../dataset"
CATEGORIES = ["with_mask", "without_mask"]
IMG_SIZE = 100  # You can change to 128 or 224 if needed

# Data containers
data = []
labels = []

# Load and preprocess images
for category in CATEGORIES:
    path = os.path.join(DATASET_PATH, category)
    class_num = CATEGORIES.index(category)  # 0 or 1
    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")

# Convert to numpy arrays
data = np.array(data) / 255.0  # Normalize
labels = to_categorical(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Data shape:", data.shape)
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))
