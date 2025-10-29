import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("EcoSave 🌿")
st.header("Image Classification: Trash vs Recycling")

# Load model
model = tf.keras.models.load_model("model/model.h5")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # match model input size
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    # Labels (match your Teachable Machine classes)
    labels = ["Trash", "Recycling"]
    st.success(f"Prediction: **{labels[class_idx]}**")
