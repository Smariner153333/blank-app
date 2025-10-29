import streamlit as st
from PIL import Image
img = Image.open(picture.png)
st.image(img, width=150) 

st.title("EcoSave")
model = tf.keras.models.load_model("keras_model.h5")

st.header("This is an Image Classification App with trash and recycling")
