import os
import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image  # Import Image class from PIL

background_image_url ='https://static.vecteezy.com/system/resources/thumbnails/026/365/937/small_2x/beautiful-blurred-green-nature-background-ai-generated-photo.jpg'

# Streamlit theme customization using CSS with a background image from URL
def set_background(image_url):
    """
    Sets the background image for the Streamlit app using CSS.
    
    Parameters:
    image_url (str): URL of the background image.
    """
    background_css = f"""
    <style>
    .reportview-container {{
        background: url("{image_url}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

# Call the function to set the background image from URL
set_background(background_image_url)

# Load the pre-trained model
model_path = r'.\braintumor.h5'
model = load_model(model_path)

st.title('Brain Tumor Classification CNN Model')

# Define class names based on your labels
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Function to preprocess and classify the image
def classify_image(image):
    # Convert image to numpy array
    img = np.array(image)

    # Resize to match model's expected sizing
    img = tf.image.resize(img, [200, 200])

    # Normalize pixel values to between 0 and 1
    img = img / 255.0  

    # Expand dimensions to match batch size used during training
    img = np.expand_dims(img, axis=0)  

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    class_name = class_names[predicted_class]

    return class_name, confidence

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Classify the image and display result
    class_name, confidence = classify_image(image)
    st.write(f"Prediction: {class_name}")
    st.write(f"Confidence: {confidence:.2f}%")
