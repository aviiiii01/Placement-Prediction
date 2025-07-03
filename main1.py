import joblib
import pandas as pd
import streamlit as st

st.title("🎓Placement Prediction Model....")
model=joblib.load("model1.pkl")
scaler=joblib.load("scaler1.pkl")
CGPA = st.number_input("Enter CGPA")
IQ = st.number_input("Enter IQ")

if st.button("Predict Placement"):
    data=pd.DataFrame([[CGPA,IQ]],columns=["cgpa",'iq'])
    sample=scaler.transform(data)
    result=model.predict(sample)
    if(result==1):
        st.write("Shabaaaash Beta Yhi Ummeed thi!!")
    else:
        st.write("Abe Padhle Saale Dhang se Kyu naak katva rha h!!")