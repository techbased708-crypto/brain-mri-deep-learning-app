import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Brain Tumor Detection App",
    page_icon="🧠",
    layout="centered"
)

# --- GOOGLE DRIVE MODEL CONFIGURATION ---
FILE_ID = '1ztn97BcVwZ9uGkmP8pbAnu1Z_skg472V'
MODEL_URL = f'https://drive.google.com/uc?id={FILE_ID}'
MODEL_PATH = 'brain_tumor_efficientnetB2_model.keras'

@st.cache_resource
def load_data_model():
    """Downloads the model from Google Drive if not present and loads it into memory."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model from Google Drive... Please wait, this may take a minute.'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            
    model = load_model(MODEL_PATH)
    return model

# Load model safely
try:
    model = load_data_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- UI DESIGN & STYLING ---
st.title("🧠 Brain Tumor Classification App")
st.markdown("Upload a brain MRI image to detect the presence and type of tumor using deep learning (EfficientNetB2).")

st.sidebar.header("About Project")
st.sidebar.info(
    "This application is a professional portfolio piece designed for medical imaging classification. "
    "It uses a fine-tuned EfficientNetB2 architecture."
)

# Class labels (Aapke model ke mutabiq classes yahan honge, agar zaroorat ho toh inhein adjust kar sakte hain)
CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_container_width=True)
    if st.button('Predict Tumor'):
        with st.spinner('Analyzing the MRI scan...'):
            try:
                # Preprocessing image for EfficientNetB2 (Standard size: 260x260 or as per your training)
                img = image.resize((260, 260))
                img_array = np.array(img)
                
                # Handle grayscale or RGBA images
                if img_array.ndim == 2:
                    img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                    
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalization if used during training
                
                # Make Prediction
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0])) * 100
                
                # Display Results
                st.success("Analysis Complete!")
                st.markdown(f"### Prediction: **{CLASSES[predicted_class_idx]}**")
                st.markdown(f"### Confidence: **{confidence:.2f}%**")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")