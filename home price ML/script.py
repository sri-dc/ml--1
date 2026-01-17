import streamlit as st
import pandas as pd
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("House Price Prediction")
st.write("Enter the details of the house to predict its price.")

area = st.number_input("Area (sq ft)", min_value=300, step=50)
bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, step=1)
location = st.selectbox("Location", ["Chennai", "Bangalore", "Hyderabad", "Pune"])

if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'location': [location]
    })
    
    predicted_price = model.predict(input_data)[0]
    st.success(f"The predicted price of the house is: ₹{predicted_price:,.2f}")