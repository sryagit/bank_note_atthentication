
import numpy as np
import pandas as pd
import joblib
import streamlit as st 
from PIL import Image

st.image(Image.open('dollar.jpg'), use_column_width=True)

def predict_note_authentication(variance, skewness, curtosis, entropy):
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    return prediction

def main():
    st.title("Banknote Authentication Classifier")
    variance = st.text_input("variance", placeholder="Type Here")
    skewness = st.text_input("skewness", placeholder="Type Here")
    curtosis = st.text_input("curtosis", placeholder="Type Here")
    entropy = st.text_input("entropy", placeholder="Type Here")

    if st.button("Get Prediction"):
        output = predict_note_authentication(variance, skewness, curtosis, entropy)
        st.success(f'Result: {output}.')
        st.write('1 = banknote is genuine, 0 = banknote is forged')

    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()
