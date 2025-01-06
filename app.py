import streamlit as st
import joblib
import os
import pytesseract
from PIL import Image
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
wave_animation = load_lottie_url("https://lottie.host/ee6589da-1fd3-4c83-aa76-3fc9c339d67f/97tqW55fBj.json")

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

# OCR function
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        st.error(f"Error during OCR: {e}")
        return None

# Main app
def main():
    # App title with animation
    if wave_animation:
        st_lottie(wave_animation, height=200, key="wave")
    else:
        st.warning("⚠️ Animation could not be loaded. Please check your network or the animation URL.")

    st.markdown("<h1 style='text-align: center; color: #4caf50;'>🕵️ Fake News Detection App</h1>", unsafe_allow_html=True)

    # Sidebar content
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRgacOEKP5hJ_z8h6lssY3an3HUnChNXP4VXg&s", width=250)
    st.sidebar.header("About the App")
    st.sidebar.info("""
    This app uses advanced machine learning techniques to detect whether a news article 
    is real or fake. Paste your article text or upload an image, and let the app classify it for you!
    """)

    # Load models
    model, vectorizer = load_models()
    if model is None or vectorizer is None:
        st.error("Failed to load models. Please ensure the files exist in 'data/models'.")
        return

    # Tabs for Text Input and OCR
    tab1, tab2 = st.tabs(["Text Input", "Image Upload (OCR)"])

    with tab1:
        st.markdown("### Enter News Text")
        user_input = st.text_area("Enter the news article text:", height=200, help="Paste the text you want to classify")
        if st.button("Classify News", key="text_btn"):
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

    with tab2:
        st.markdown("### Upload an Image for OCR")
        uploaded_image = st.file_uploader("Upload an image containing text:", type=["png", "jpg", "jpeg"])
        if st.button("Extract and Classify", key="ocr_btn"):
            if uploaded_image is not None:
                try:
                    image = Image.open(uploaded_image)
                    extracted_text = extract_text_from_image(image)
                    if extracted_text:
                        st.markdown("#### Extracted Text")
                        st.write(extracted_text)

                        # Classify the extracted text
                        result, confidence = classify_news(extracted_text, model, vectorizer)
                        if result == "Real News":
                            st.success(f"🟢 **The article is classified as: {result}**")
                        else:
                            st.error(f"🔴 **The article is classified as: {result}**")
                        st.info(f"Confidence: {confidence:.2f}%")
                    else:
                        st.warning("No text could be extracted from the image.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please upload an image.")

    # Footer with a custom message
    st.markdown("<h3 style='text-align: center; color: #4caf50;'>The Keyboard Crackers 🌟</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
