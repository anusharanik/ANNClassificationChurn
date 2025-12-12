import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#Load the trained model
model=tf.keras.models.load_model('modelRegression.h5')

with open('labelEncoderGenderRegression.pkl','rb') as file:
    labelEncoderGender=pickle.load(file)
with open('onehotEncoderGeoRegression.pkl','rb')as file:
    onehotEncoderGeo=pickle.load(file)
with open('scalerRegression.pkl','rb') as file:
    scaler=pickle.load(file)

##Streamlit app
st.title('Estimated Salary Prediction')

#User input
geography = st.selectbox('Geography', onehotEncoderGeo.categories_[0])
gender = st.selectbox('Gender', labelEncoderGender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [labelEncoderGender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode 'Geography'
geo_encoded = onehotEncoderGeo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotEncoderGeo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

#Predict estimated salary
prediction=model.predict(input_data_scaled)
preddiction_salary=prediction[0][0]

st.write(f'Predicted Estimated Salary: ${preddiction_salary:.2f}')

