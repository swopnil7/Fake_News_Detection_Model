import streamlit as st
import joblib
import os

# Ensure the models directory exists
os.makedirs('data/models', exist_ok=True)

def load_models():
    """
    Load the pre-trained model and vectorizer.
    """
    try:
        model = joblib.load('data/models/logistic_regression_model.joblib')
        vectorizer = joblib.load('data/models/tfidf_vectorizer.joblib')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def classify_news(text, model, vectorizer):
    """
    Classify a piece of text as real or fake news.
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

    # Load pre-trained model and vectorizer
    model, vectorizer = load_models()
    if model is None or vectorizer is None:
        st.error("Failed to load models. Please ensure the files exist in 'data/models'.")
        return

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
