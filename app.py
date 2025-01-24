import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    size = (150, 150)    
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)

    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    
    return prediction

model = tf.keras.models.load_model('C:/Users/Aidan/Important/MLAI_Project/MLAI_Project/Models/model02_theyan.h5') #loading a trained model

st.write("""
         # Bao & Uni Sushi Predictor
         """
)

st.write("This is a simple image classification web app to predict bao and uni sushi")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is bao")
        # Show link for nearby bao restaurants
        st.markdown('[Search for nearby bao restaurants](https://www.google.com/maps/search/bao+restaurants+near+me)', unsafe_allow_html=True)
    elif np.argmax(prediction) == 1:
        st.write("It is neither")
    else:
        st.write("It is uni sushi")
        # Show link for nearby sushi restaurants
        st.markdown('[Search for nearby sushi restaurants](https://www.google.com/maps/search/uni+sushi+restaurants+near+me)', unsafe_allow_html=True)
    
    st.text("Probability (0: bao, 1: neither, 2: uni sushi)")
    st.write(prediction)
