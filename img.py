import streamlit as st
from PIL import Image
import numpy as np
import pickle
import requests
from io import BytesIO

# Load the model
with open('model02.pkl', 'rb') as f:
    model = pickle.load(f)
  #  st.write(model)
    
st.title(" Facial Expression Recognition")
st.write("Facial expression recognition is the task of classifying the expressions on face images into various categories such as anger, fear, surprise, sadness, happiness and so on. Emotional facial expressions can inform researchers about an individual's emotional state.")

Emotion_Classes = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def classify_image(image, model):
    img = image.resize((48, 48))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction


uploaded_files = st.file_uploader("Choose an Image...", type=["jpg"], accept_multiple_files=True)
image_url = st.text_input("Or Enter Image URL")

if uploaded_files is not None:
    if len(uploaded_files) > 0:
        cols = st.columns(len(uploaded_files))

        for col, uploaded_file in zip(cols, uploaded_files):
            image = Image.open(uploaded_file).convert('L')

            with col:
                prediction = classify_image(image, model)
                st.write(f'{Emotion_Classes[np.argmax(prediction)]}: {100*np.max(prediction):2.0f}%')
                st.image(image, width=100)

# Display the image from the URL
if image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('L')
        prediction = classify_image(image, model)
        st.write(f'{Emotion_Classes[np.argmax(prediction)]}: {100*np.max(prediction):2.0f}%')
        st.image(image, width=100)
    except:
        st.error("Invalid URL or unable to fetch the image.")