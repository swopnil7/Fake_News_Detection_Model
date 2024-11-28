import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Ensure the models directory exists
os.makedirs('data/models', exist_ok=True)

def load_data():
    try:
        # Load the dataset
        data = pd.read_csv('data/processed/processed_dataset.csv')
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def train_model(data):
    """
    Train a logistic regression model for fake news detection
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )

    # Vectorize the text
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectorized, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model and vectorizer
    joblib.dump(model, 'data/models/logistic_regression_model.joblib')
    joblib.dump(vectorizer, 'data/models/tfidf_vectorizer.joblib')

    return model, vectorizer

def classify_news(text, model, vectorizer):
    """
    Classify a piece of text as real or fake news
    """
    # Vectorize the input text
    vectorized_input = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(vectorized_input)
    proba = model.predict_proba(vectorized_input)[0]
    
    # Determine result with probability
    result = "Real News" if prediction[0] == 1 else "Fake News"
    confidence = proba[prediction[0]] * 100
    
    return result, confidence

def main():
    st.title("🕵️ Fake News Detection App")
    
    # Check if model exists, if not train it
    if not (os.path.exists('data/models/logistic_regression_model.joblib') and 
            os.path.exists('data/models/tfidf_vectorizer.joblib')):
        st.warning("No pre-trained model found. Training a new model...")
        data = load_data()
        if data is not None:
            model, vectorizer = train_model(data)
        else:
            st.error("Dataset loading failed. Cannot proceed with training.")
            return
    else:
        # Load existing model
        model = joblib.load('data/models/logistic_regression_model.joblib')
        vectorizer = joblib.load('data/models/tfidf_vectorizer.joblib')

    # User input section
    st.header("News Article Classification")
    user_input = st.text_area("Enter the news article text:", 
                               height=200, 
                               help="Paste the text you want to classify")

    if st.button("Classify News"):
        if user_input.strip():
            try:
                # Classify the news
                result, confidence = classify_news(user_input, model, vectorizer)
                
                # Display results
                if result == "Real News":
                    st.success(f"🟢 The article is classified as: {result}")
                else:
                    st.error(f"🔴 The article is classified as: {result}")
                
                st.info(f"Confidence: {confidence:.2f}%")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to classify.")

    # Optional: Add some additional information
    st.sidebar.header("About the App")
    st.sidebar.info("""
    This Fake News Detection app uses machine learning to 
    classify news articles as real or fake. 
    
    Note: The accuracy depends on the training data and model.
    """)

if __name__ == "__main__":
    main()
