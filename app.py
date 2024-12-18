import streamlit as st
import joblib
import os
from streamlit_lottie import st_lottie
import requests

def load_lottie_url(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception as e:
        st.error(f"Error loading animation: {e}")
        return None

wave_animation = load_lottie_url("https://lottie.host/ee6589da-1fd3-4c83-aa76-3fc9c339d67f/97tqW55fBj.json")

def load_models():
    try:
        model = joblib.load('data/models/logistic_regression_model.joblib')
        vectorizer = joblib.load('data/models/tfidf_vectorizer.joblib')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def classify_news(text, model, vectorizer):
    vectorized_input = vectorizer.transform([text])
    prediction = model.predict(vectorized_input)
    proba = model.predict_proba(vectorized_input)[0]
    result = "Real News" if prediction[0] == 1 else "Fake News"
    confidence = proba[prediction[0]] * 100
    return result, confidence

def main():
    if wave_animation:
        st_lottie(wave_animation, height=200, key="wave")
    else:
        st.warning("⚠️ Animation could not be loaded. Please check your network or the animation URL.")

    st.markdown("<h1 style='text-align: center; color: #4caf50;'>🕵️ Fake News Detection App</h1>", unsafe_allow_html=True)

    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRgacOEKP5hJ_z8h6lssY3an3HUnChNXP4VXg&s", width=250)
    st.sidebar.header("About the App")
    st.sidebar.info("""
    This app uses advanced machine learning techniques to detect whether a news article 
    is real or fake. Simply paste your article text, and let the app classify it for you!
    """)

    model, vectorizer = load_models()
    if model is None or vectorizer is None:
        st.error("Failed to load models. Please ensure the files exist in 'data/models'.")
        return

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

    user_input = st.text_area("Enter the news article text:", height=200, help="Paste the text you want to classify")
    if st.button("Classify News"):
        if user_input.strip():
            try:
                result, confidence = classify_news(user_input, model, vectorizer)
                if result == "Real News":
                    st.success(f"🟢 **The article is classified as: {result}**")
                else:
                    st.error(f"🔴 **The article is classified as: {result}**")
                st.info(f"Confidence: {confidence:.2f}%")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to classify.")

    st.markdown("<h3 style='text-align: center; color: #4caf50;'>The Keyboard Crackers 🌟</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
