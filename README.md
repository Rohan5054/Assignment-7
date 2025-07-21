This project is a House Price Prediction system built using machine learning. It consists of two main parts: the training script (train_model.py) and a Streamlit web app (app.py). The training script processes the dataset, trains multiple regression models, and saves the best model along with the feature scaler. The Streamlit app provides a user-friendly interface to input house features and predicts the house price using the saved model.

Getting Started
Train the Model: First, run train_model.py to preprocess the data, train the models, and save the best model and scaler in the model/ directory:

python train_model.py

Run the Web App: Once the model is trained, run the app.py to launch the web interface:

bash
Copy
Edit
streamlit run app.py
The app allows users to input features like Overall Quality, Garage Capacity, etc., and it will predict the house price.
