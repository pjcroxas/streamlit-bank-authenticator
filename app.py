import numpy as np
import pickle
import pandas as pd
import streamlit as st

model_ = open("classifier.pkl", "rb")
classifier=pickle.load(model_)

def welcome():
    return "Welcome All"

def predict_note_authenticator(variance, skewness, kurtosis, entropy):
    prediction = classifier.predict([[variance,
                                      skewness,
                                      kurtosis,
                                      entropy]])
    print(prediction)
    return prediction

def main():
    st.title("Bank Authenticator")
    html_temp = """"
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>   
    
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    variance = st.text_input("Variance", "Type Here")
    skewness = st.text_input("Skewness", "Type Here")
    kurtosis = st.text_input("Kurtosis", "Type Here")
    entropy = st.text_input("Entropy", "Type Here")
    result = ""
    if st.button("Predict"):
        result=predict_note_authenticator(variance, skewness, kurtosis, entropy)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Let's Learn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()



