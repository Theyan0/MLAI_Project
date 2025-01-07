import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# Function to preprocess the image and make predictions
def import_and_predict(image_data, model):
    size = (150, 150)  # Resize to match the model's expected input size
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)  # Normalize pixel values

    img_reshape = image[np.newaxis, ...]  # Add batch dimension

    prediction = model.predict(img_reshape)  # Predict the class probabilities
    return prediction

# Load the trained model
model = tf.keras.models.load_model('C:/Users/sunda/OneDrive/Documents/ML&AI LABS/MLAI_Project/Models/model01_theyan.h5')

# Streamlit app title and description
st.write("""
         # Bao, Random, Uni Sushi Classification
         """
         )
st.write("This is a simple image classification web app to predict whether the image is Bao, Random, or Uni Sushi.")

# File uploader for image input
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Make prediction
    prediction = import_and_predict(image, model)
    class_names = ['Bao', 'Random', 'Uni Sushi']  # Class labels corresponding to the model's output

    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"The image is classified as: **{predicted_class}**")
    
    st.text("Probability (0: Bao, 1: Random, 2: Uni Sushi)")
    st.write(prediction)
