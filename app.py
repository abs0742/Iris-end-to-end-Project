# Import required packages
import pandas as pd
import pickle
import streamlit as st
import numpy as np

# Create the header for brower
st.set_page_config(page_title='Iris Project - Abhijeet Sharma')

# Add a title for the body
st.title("Iris End-to-End Project - Abhijeet Sharma")

# Take sepal lengh, sepal width, petal length, petal width
sep_len = st.number_input('Sepal Length : ', min_value=0.00, step=0.1)
sep_wid = st.number_input('Sepal Width : ', min_value=0.00, step=0.1)
pet_len = st.number_input('Petal Length : ', min_value=0.00, step=0.1)
pet_wid = st.number_input('Petal Width : ', min_value=0.00, step=0.1)

# Add a button to predict the result
Submit = st.button('Predict')

#Write a function to predict species along with probability
def predict_species(pre_path, model_path):
    # Get the inputes in dataframe format
    xnew = pd.DataFrame([sep_len, sep_wid, pet_len, pet_wid]).T
    xnew.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    # Load the preprocessor with pickle
    with open(pre_path, 'rb') as file1:
        pre = pickle.load(file1)
    # Transform xnew
    xnew_pre = pre.transform(xnew)
    # Load the model with pickle
    with open(model_path, 'rb') as file2:
        model = pickle.load(file2)
    # Get the predictions
    preds = model.predict(xnew_pre)
    # Get the probability
    probs = model.predict_proba(xnew_pre)
    # Get the maximum probability
    max_prob = np.max(probs)
    return preds, max_prob

# Subheader to show results
st.subheader('Result are : ')

# Predicting result on the web app after submit button is pressed
if Submit:
    # Get pre path and model path
    pre_path = 'notebook/preprocessor.pkl'
    model_path = 'notebook/model.pkl'
    # Get the predictions along with probability
    pred, max_prob = predict_species(pre_path, model_path)
    # Print the result
    st.subheader(f'Predicted Species are : {pred[0]}')
    st.subheader(f'Probability of Prediction : {max_prob:.4f}')
    # Show probabilty in progress bar
    st.progress(max_prob)