import base64
import streamlit as st
from st_clickable_images import clickable_images
import pandas as pd
import numpy as np
from PIL import Image
from fastai.vision.all import load_learner

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

categories = ('hotdog', 'pasta', 'pizza', 'salad', 'sandwich')

st.write(""" 
         # Simple streamlit app for hotdog classification

          """)
st.write(f'Categories: {categories}')
st.sidebar.header('User image upload')

def classify_images(img):
    learn = load_learner('hotdog.pkl')
    pred, idx, probs = learn.predict(img)
    # st.write( dict(zip(categories, map(float, probs)))))
    return categories[np.argmax(probs)]

def user_input(): 
    images = st.sidebar.file_uploader('load images to classify', 
                             type=['png', 'jpg'],
                            accept_multiple_files=True )
    
    return images 

uploaded_files = user_input()

st.sidebar.header('Example images')

# Create a clickable list in the sidebar
clicked_ex = st.sidebar.selectbox(
    "Click to select an example",
     [None]+[c for c in categories],

)

# Display clicked example image
if clicked_ex is not None :
    st.sidebar.markdown(f"### Selected Example")
    img_ex = np.array(Image.open(f'JH/{clicked_ex}.jpg'))
    st.sidebar.image(img_ex, caption=f"Image #{clicked_ex}")

    prediction = classify_images(img_ex)
    print(prediction)

    st.write(f"prediction {prediction}")

for uploaded_img in uploaded_files:
    st.sidebar.write(f"file {uploaded_img.name} loaded")
    img = np.array(Image.open(uploaded_img))
    st.sidebar.image(img, caption='Input image')

    prediction = classify_images(img)
    print(prediction)

    st.write(f"prediction {prediction}")



