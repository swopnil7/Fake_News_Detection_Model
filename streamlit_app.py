import streamlit as st
import pickle
import pandas as pd

# Load your machine learning model
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.title("Fake News Detection App")

user_input = st.text_area("Enter the news article text:")

if st.button("Classify"):
    if user_input.strip():
        # Preprocess and vectorize the input
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)
        result = "Real News" if prediction[0] == 1 else "Fake News"
        st.success(f"The article is classified as: {result}")
    else:
        st.error("Please enter some text to classify.")
