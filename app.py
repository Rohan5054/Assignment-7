import streamlit as st
import pickle
import numpy as np
import os

# Print the current working directory to check where the app is running from
print("Current working directory:", os.getcwd())

# Define the paths for model, scaler, and kmeans files
model_file = os.path.join(os.getcwd(), 'model', 'best_house_price_model.pkl')  # Updated model name
scaler_file = os.path.join(os.getcwd(), 'model', 'scaler.pkl')
kmeans_file = os.path.join(os.getcwd(), 'model', 'kmeans.pkl')

# Debug: Check if the model file exists
if not os.path.exists(model_file):
    st.error(f"Model file not found at {model_file}")
    print(f"Model file not found at {model_file}")  # Debugging
else:
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            print("Model loaded successfully.")  # Debugging
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        print(f"Error loading model: {str(e)}")  # Debugging

# Debug: Check if the scaler file exists
if not os.path.exists(scaler_file):
    st.error(f"Scaler file not found at {scaler_file}")
    print(f"Scaler file not found at {scaler_file}")  # Debugging
else:
    try:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
            print("Scaler loaded successfully.")  # Debugging
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        print(f"Error loading scaler: {str(e)}")  # Debugging

# Debug: Check if the kmeans file exists
if not os.path.exists(kmeans_file):
    st.error(f"KMeans file not found at {kmeans_file}")
    print(f"KMeans file not found at {kmeans_file}")  # Debugging
else:
    try:
        with open(kmeans_file, 'rb') as f:
            kmeans = pickle.load(f)
            print("KMeans loaded successfully.")  # Debugging
    except Exception as e:
        st.error(f"Error loading KMeans: {str(e)}")
        print(f"Error loading KMeans: {str(e)}")  # Debugging

# Set up the Streamlit app
st.title("House Price Prediction")
st.subheader("Enter the features to predict the house price:")

# Define input fields based on the selected top features
OverallQual = st.slider("Overall Quality (1=Very Poor, 10=Excellent)", 1, 10, 5)
GrLivArea = st.number_input("Above Ground Living Area (in sq ft)", min_value=200, max_value=5000, step=10)
GarageCars = st.slider("Garage Capacity (Number of Cars)", 0, 4, 2)
TotalBsmtSF = st.number_input("Total Basement Area (in sq ft)", min_value=0, max_value=3000, step=10)
YearBuilt = st.number_input("Year Built", min_value=1900, max_value=2022, step=1)
TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 1, 15, 6)
Fireplaces = st.slider("Number of Fireplaces", 0, 3, 1)

# When the user clicks "Predict", make the prediction
if st.button("Predict"):
    # Check if model, scaler, and kmeans are loaded properly
    if 'model' not in locals() or 'scaler' not in locals() or 'kmeans' not in locals():
        st.error("Model, scaler, or KMeans not loaded properly. Please check the logs.")
        print("Model, scaler, or KMeans not loaded properly.")  # Debugging
    else:
        # Prepare the input array for prediction
        inputs = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, YearBuilt, TotRmsAbvGrd, Fireplaces]])

        # Scale the input using the loaded scaler
        inputs_scaled = scaler.transform(inputs)

        # Apply KMeans clustering to the input data (same transformation as training)
        cluster = kmeans.predict(inputs_scaled)
        inputs_scaled_with_cluster = np.hstack([inputs_scaled, cluster.reshape(-1, 1)])

        # Make prediction using the loaded model
        prediction = model.predict(inputs_scaled_with_cluster)

        # Show the prediction result
        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
