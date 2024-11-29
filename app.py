import streamlit as st
import joblib
import os
from streamlit_lottie import st_lottie
import requests

# Load Lottie animation
def load_lottie_url(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception as e:
        st.error(f"Error loading animation: {e}")
        return None

# Load animations
wave_animation = load_lottie_url("https://lottie.host/2aea07e1-5cfd-45a6-9f37-bff928831f92/rFSSBoTMGw.json")

# Load models
def load_models():
    try:
        model = joblib.load('data/models/logistic_regression_model.joblib')
        vectorizer = joblib.load('data/models/tfidf_vectorizer.joblib')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Classification function
def classify_news(text, model, vectorizer):
    vectorized_input = vectorizer.transform([text])
    prediction = model.predict(vectorized_input)
    proba = model.predict_proba(vectorized_input)[0]
    result = "Real News" if prediction[0] == 1 else "Fake News"
    confidence = proba[prediction[0]] * 100
    return result, confidence

# Main app
def main():
    # App title with animation
    if wave_animation:
        st_lottie(wave_animation, height=200, key="wave")
    else:
        st.warning("⚠️ Animation could not be loaded. Please check your network or the animation URL.")

    st.markdown("<h1 style='text-align: center; color: #4caf50;'>🕵️ Fake News Detection App</h1>", unsafe_allow_html=True)

    # Sidebar content
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/Fake_News.png", width=250)
    st.sidebar.header("About the App")
    st.sidebar.info("""
    This app uses advanced machine learning techniques to detect whether a news article 
    is real or fake. Simply paste your article text, and let the app classify it for you!
    """)

    # Load models
    model, vectorizer = load_models()
    if model is None or vectorizer is None:
        st.error("Failed to load models. Please ensure the files exist in 'data/models'.")
        return

    # Input section with enhanced styles
    st.markdown(
        """
        <style>
        .stTextArea>div>div>textarea {
            border: 2px solid #4caf50;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            color: #333;
        }
        .stButton>button {
            background-color: #4caf50;
            color: white;
            border-radius: 5px;
            font-size: 16px;
            padding: 10px;
            margin-top: 20px;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # User input and classification
    user_input = st.text_area("Enter the news article text:", height=200, help="Paste the text you want to classify")
    if st.button("Classify News"):
        if user_input.strip():
            try:
                result, confidence = classify_news(user_input, model, vectorizer)
                # Display results with styled messages
                if result == "Real News":
                    st.success(f"🟢 **The article is classified as: {result}**")
                else:
                    st.error(f"🔴 **The article is classified as: {result}**")
                st.info(f"Confidence: {confidence:.2f}%")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to classify.")

    # Footer with a custom message
    st.markdown("<h3 style='text-align: center; color: #4caf50;'>Powered by Machine Learning 🌟</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
