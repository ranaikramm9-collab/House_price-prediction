# 🏠 House Price Prediction App

This project is a Machine Learning-based House Price Prediction system that predicts property prices using multiple features such as living area, bathrooms, bedrooms, lot size, location, and derived features. The model is trained on housing data from Seattle and surrounding cities in Washington State, USA, including areas like Seattle, Bellevue, Redmond, Kirkland, Renton, and others.

The application includes both a **Streamlit frontend** for interactive user input and a **FastAPI backend** for serving predictions through an API. The model uses encoded categorical features for city and statezip, along with numerical features like sqft_living, sqft_above, sqft_lot, bathrooms, bedrooms, house_age, and sqft_per_bedroom to generate accurate price predictions.

The project demonstrates a complete end-to-end machine learning pipeline including data preprocessing, feature engineering, model training, saving/loading the model using joblib, and deploying it as a web application.

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Streamlit, FastAPI, Joblib

## How to Run
Streamlit App:  
streamlit run app.py  

FastAPI Backend:  
uvicorn main:app --reload  

## Dataset Region
Seattle and nearby cities in Washington State, USA

This project is built for educational purposes and demonstrates real-world ML deployment workflow from training to production.