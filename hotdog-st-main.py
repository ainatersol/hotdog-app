import base64
import streamlit as st
from st_clickable_images import clickable_images
import pandas as pd
import numpy as np
from PIL import Image
from fastai.vision.all import *

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


learn = load_learner('hotdog.pkl')
categories = ('hotdog', 'pasta', 'pizza', 'salad', 'sandwich')

st.write(""" 
         # Simple streamlit app for hotdog classification

          """)

st.sidebar.header('User image upload')

def user_input(): 
    images = st.sidebar.file_uploader('load images to classify', 
                             type=['png', 'jpg'],
                            accept_multiple_files=True )
    
    return images 

uploaded_files = user_input()


st.sidebar.header('Example images')
example_images = [get_image_base64(f'JH/{c}.jpg') for c in categories]
clicked = clickable_images(
    [
        f"data:image/jpeg;base64,{img_base64}"
        for img_base64 in example_images
    ],
    titles=[f"Image #{c}" for c in categories],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "200px"},
)

st.sidebar.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")

st.write(f'Categories: {categories}')

def classify_images(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


for uploaded_img in uploaded_files:
    bytes_data = uploaded_img.read()
    st.sidebar.write(f"file {uploaded_img.name} loaded")
    img = np.array(Image.open(uploaded_img))
    st.image(img, caption='Input image')

    prediction = classify_images(img)
    print(prediction)

    st.write(f"prediction {prediction}")



