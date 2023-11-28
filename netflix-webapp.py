# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/admino/Documents/Model Deployment/trained_model.sav', 'rb'))

#Loading thr scaler
loaded_scaler = pickle.load(open('C:/Users/admino/Documents/Model Deployment/scaler.pkl', 'rb'))

st.title('Netflix Stock Price Prediction')

# Create a form to collect user input
Open = st.number_input('Open Price')
High = st.number_input('High Price')
Low = st.number_input('Low Price')
Adj_Close = st.number_input('Adjusted Close Price')
Volume = st.number_input('Volume')
year = st.number_input('Year')
month = st.number_input('Month')
day = st.number_input('Day')

if st.button('Predict'):
    # Prepare the input data
    new_data = np.array([[Open, High, Low, Adj_Close, Volume, year, month, day]])

    # Scale the input data
    new_data_scaled = loaded_scaler.transform(new_data)

    # Make the prediction
    prediction = loaded_model.predict(new_data_scaled)[0]

    # Display the prediction
    st.write('Predicted Close Price:', prediction)