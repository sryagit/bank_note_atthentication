import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image

model = joblib.load('classifier.joblib')

image = Image.open('dollar.png')
st.image(image.resize((1000, 300)))

def predict_note_authentication(variance, skewness, curtosis, entropy):
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    return prediction

def main():
    st.title("Bank Note Authentication Web APP")
    variance = st.text_input("Variance", placeholder="Type Here")
    skewness = st.text_input("Skewness", placeholder="Type Here")
    curtosis = st.text_input("Curtosis", placeholder="Type Here")
    entropy = st.text_input("Entropy", placeholder="Type Here")

    if st.button("Get Prediction"):
        output = predict_note_authentication(variance, skewness, curtosis, entropy)
        if output == 0:
            st.markdown("<h3>Result :<span style='color:red'> 0 </span></h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3>Result :<span style='color:green'> 1 </span></h3>", unsafe_allow_html=True)
            
        st.markdown("<p style='color:red'> 0 = banknote is forged </p>",  unsafe_allow_html=True)
        st.markdown("<p style='color:green'> 1 = banknote is genuine</p>", unsafe_allow_html=True)
        st.text("Classifier : Random Forest")
        st.text("Accuracy : 99.27 %")
        st.text("Built by : Suraj R. Yadav")

if __name__ == '__main__':
    main()
