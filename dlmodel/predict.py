# predict.py
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pdf2image import convert_from_path
from PIL import Image

# --- CONFIG ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 10  # adjust as needed
BATCH_SIZE = 1
THRESHOLD = 0.01  # reconstruction error threshold

# --- PDF / IMAGE PREPROCESSING FUNCTIONS ---
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def convert_pdf_to_image(pdf_path, page=0):
    images = convert_from_path(pdf_path)
    img = images[page].convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- UNZIP uploaded dataset ---
zip_path = "receipt98.zip"
extract_path = "receipt98"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print(f"✅ Extracted to {extract_path}")

# --- LOAD IMAGES FROM FOLDER ---
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)

X = load_images_from_folder(extract_path)
print(f"✅ Loaded {len(X)} images")

# --- BUILD AUTOENCODER ---
def build_autoencoder():
    input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Encoder
    x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2,2), padding='same')(x)

    # Decoder
    x = Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    decoded = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

autoencoder = build_autoencoder()

# --- TRAIN AUTOENCODER ---
history = autoencoder.fit(X, X, epochs=EPOCHS, batch_size=BATCH_SIZE)
print("✅ Final training loss:", history.history['loss'][-1])

# --- PLOT SAMPLE RECONSTRUCTION ---
decoded_img = autoencoder.predict(np.expand_dims(X[0], axis=0))[0]

plt.subplot(1,2,1)
plt.imshow(X[0])
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(decoded_img)
plt.title("Reconstructed")
plt.axis("off")
plt.show()

# --- FUNCTION TO CALCULATE RECONSTRUCTION ERROR ---
def reconstruction_error(original, reconstructed):
    return mean_squared_error(original.flatten(), reconstructed.flatten())

# --- TEST ON IMAGE ---
test_image_path = "morebill.jpg"  # change to your test image
test_img = preprocess_image(test_image_path)
reconstructed_img = autoencoder.predict(test_img)[0]
error = reconstruction_error(test_img[0], reconstructed_img)

if error < THRESHOLD:
    print(f"✅ This bill is likely REAL (error={error:.6f})")
else:
    print(f"❌ This bill is likely FAKE (error={error:.6f})")

# --- TEST ON PDF ---
pdf_test_path = "invoice.pdf"  # change to your test PDF
test_img_pdf = convert_pdf_to_image(pdf_test_path)
reconstructed_pdf = autoencoder.predict(test_img_pdf)[0]
error_pdf = reconstruction_error(test_img_pdf[0], reconstructed_pdf)

if error_pdf < THRESHOLD:
    print(f"✅ REAL receipt (error: {error_pdf:.6f})")
else:
    print(f"❌ FAKE receipt (error: {error_pdf:.6f})")