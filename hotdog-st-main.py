import streamlit as st
import pandas as pd
import numpy as np

from fastai.vision.all import *


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
for uploaded_img in uploaded_files:
    bytes_data = uploaded_img.read()
    st.write("filename:", uploaded_img.name)
    st.write(bytes_data)

learn = load_learner('hotdog.pkl')
categories = ('hotdog', 'pasta', 'pizza', 'salad', 'sandwich')

def classify_images(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))