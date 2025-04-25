import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib


kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')

# Streamlit App UI
st.title("Credit Risk Cluster Predictor")

st.write("Enter the applicant's details:")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", encoders['Sex'].classes_)
job = st.slider("Job (0 = unemployed, 3 = highly skilled)", 0, 3, 1)
housing = st.selectbox("Housing", encoders['Housing'].classes_)
saving_acc = st.selectbox("Saving accounts", encoders['Saving accounts'].classes_)
checking_acc = st.selectbox("Checking account", encoders['Checking account'].classes_)
credit_amount = st.number_input("Credit Amount", min_value=0, step=100, value=1000)
duration = st.number_input("Duration (months)", min_value=1, step=1, value=12)
purpose = st.selectbox("Purpose", encoders['Purpose'].classes_)


if st.button("Predict Cluster"):
   
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [encoders['Sex'].transform([sex])[0]],
        'Job': [job],
        'Housing': [encoders['Housing'].transform([housing])[0]],
        'Saving accounts': [encoders['Saving accounts'].transform([saving_acc])[0]],
        'Checking account': [encoders['Checking account'].transform([checking_acc])[0]],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [encoders['Purpose'].transform([purpose])[0]]
    })

    
    input_data['Unnamed: 0'] = 0  

    
    input_data = input_data[['Unnamed: 0', 'Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']]

    
    input_scaled = scaler.transform(input_data)

   
    cluster = kmeans.predict(input_scaled)[0]
    
    
    st.success(f"This applicant belongs to Cluster #{cluster}")

    
    cluster_desc = {
        0: "🔵 Likely low-risk: financially stable group",
        1: "🟡 Medium-risk: average credit pattern",
        2: "🔴 High-risk: applicants with challenging credit traits"
    }

    st.write(cluster_desc.get(cluster, "No description available."))
