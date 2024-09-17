import streamlit as st
import pickle
import numpy as np

# Load the saved churn prediction model
with open(r'C:\Users\ezer2\PycharmProjects\Project1\churn.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app interface
st.title('Customer Churn Prediction')

st.write("""
### Enter customer data to predict churn:
""")

# Input fields for user to enter customer information
tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, step=1)
montant = st.number_input('Montant (amount spent)', min_value=0.0, max_value=100000.0, step=100.0)
frequence_rech = st.number_input('Frequency of Recharge', min_value=0.0, max_value=100.0, step=1.0)
data_volume = st.number_input('Data Volume (MB)', min_value=0.0, max_value=100000.0, step=100.0)
arpu_segment = st.number_input('ARPU Segment (average revenue per user)', min_value=0.0, max_value=10000.0, step=100.0)

# Button to trigger prediction
if st.button('Predict Churn'):
    # Prepare the input as a 2D array for the model
    customer_data = np.array([[tenure, montant, frequence_rech, data_volume, arpu_segment]])

    # Make the prediction using the loaded model
    prediction = model.predict(customer_data)

    # Output the prediction (assuming 1 = churn, 0 = no churn)
    if prediction[0] == 1:
        st.write("**This customer is likely to churn.**")
    else:
        st.write("**This customer is unlikely to churn.**")
