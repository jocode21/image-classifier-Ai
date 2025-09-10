import os
os.environ["STREAMLIT_CONFIG_DIR"] = ".streamlit"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
model = tf.keras.models.load_model("best_model.h5")

st.title("🐶🐱 Image Classifier")
st.write("Upload an image of a cat or dog, and I'll tell you which one it is!")

uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    x = np.array(img) / 255.0
    x = np.expand_dims(x, 0)

    prediction = model.predict(x)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    st.subheader(f"Prediction: {label} ({prediction:.2f})")
