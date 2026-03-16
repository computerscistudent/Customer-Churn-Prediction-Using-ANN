import pandas as pd
import streamlit as st 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import numpy as np

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)


st.title('Customer Churn Prediction with ANN')

geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balace')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
number_of_products = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],   # Changed from 'Credit Score'
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products], # Changed from 'Number of Products'
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0], # Mapping directly here is cleaner
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0], # Changed from 'Is Active Member'
    'EstimatedSalary': [estimated_salary] # Changed from 'Estimated Salary'
})

input_data['Gender'] = label_encoder.transform(input_data['Gender'])

geography_encoded = onehot_encoder.transform(input_data[['Geography']])
geo_data = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography'])) 

input_data = pd.concat([input_data.drop('Geography', axis=1), geo_data], axis=1)

input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    churn_prediction = prediction[0][0]
    churn_prediction = float(round(churn_prediction, 2)*100)
    if churn_prediction >= 50:
        st.error(f'The customer is likely to churn with a probability of {churn_prediction}%.')
    else:        
        st.success(f'The customer is unlikely to churn with a probability of {churn_prediction}%.')

st.write('Churn prediction is based on the input features provided. Please ensure that the data is accurate for better predictions.')        

