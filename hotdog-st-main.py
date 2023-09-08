import streamlit as st
import pandas as pd
import numpy as np

from fastai.vision.all import *

learn = load_learner('hotdog.pkl')
categories = ('hotdog', 'pasta', 'pizza', 'salad', 'sandwich')

def classify_images(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

st.write("#Simple streamlit app for hotdog classification")
