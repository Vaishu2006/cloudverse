import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import subprocess
import sys

# --- Ensure pdf2image + poppler are installed ---
try:
    from pdf2image import convert_from_path
except ImportError:
    print("⚠️ pdf2image not found. Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdf2image"])
    try:
        # Install poppler if on Linux (Colab/Ubuntu)
        subprocess.check_call(["apt-get", "install", "-y", "poppler-utils"])
    except Exception as e:
        print("⚠️ Could not auto-install poppler. Please install manually if needed.")
    from pdf2image import convert_from_path  # retry import

IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # shape = (1, 128, 128, 3)

def convert_pdf_to_image(pdf_path, page=0):
    images = convert_from_path(pdf_path)
    img = images[page].convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # shape = (1, 128, 128, 3)