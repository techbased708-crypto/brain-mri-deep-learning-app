import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Page Configuration
st.set_page_config(page_title="Brain Tumor Detection App", page_icon="🧠", layout="centered")

# Model Load karne ka function (Cache taaki bar-bar load na ho)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('brain_tumor_efficientnetB2_model.keras')
    return model

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload a brain MRI scan image below to classify the tumor type using **EfficientNetB2**.")

# File Uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
    
    if st.button('Predict Tumor Type'):
        with st.spinner('Analyzing the scan...'):
            # Image preprocessing
            img = image.resize((224, 224))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Prediction
            preds = model.predict(x)
            pred_idx = np.argmax(preds[0])
            pred_class = class_names[pred_idx]
            confidence = np.max(preds[0]) * 100
            
            # Results Display
            st.markdown("---")
            st.subheader("Prediction Results")
            st.success(f"**Detected Tumor Type:** {pred_class.upper()}")
            st.info(f"**Confidence Score:** {confidence:.2f}%")