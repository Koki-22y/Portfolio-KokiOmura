import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import random

# Load the trained model
model = load_model('hiragana_model.h5')

# Function to preprocess input image
def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values
    return img

# Function to predict the label of input image
def predict_hiragana(image_path, model, label_map):
    target_size = (32, 32)
    img = preprocess_image(image_path, target_size)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_label_index = np.argmax(prediction)
    predicted_label = label_map[predicted_label_index]
    return predicted_label

# Folder containing test images
test_image_folder = '/Users/oomurawataruki/Downloads/HiraganaClassifier/data/test'

# Get list of files in the folder
image_files = os.listdir(test_image_folder)

# Randomly select 20 images
random_images = random.sample(image_files, 3)

# Define label map
hiragana_list = [
    "kanaA", "kanaE", "kanaI", "kanaO", "kanaU"]
label_map = {i: hiragana_list[i] for i in range(len(hiragana_list))}

# Predict and print results
for image_file in random_images:
    image_path = os.path.join(test_image_folder, image_file)
    predicted_hiragana = predict_hiragana(image_path, model, label_map)
    print(f"Predicted Hiragana for {image_file}: {predicted_hiragana}")
