#train_model.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
import sys

def load_data(file_path):
    # Check if the specified file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    try:
        # Load the dataset from a CSV file
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Drop any rows with missing values
    df = df.dropna()
    # Ensure that required columns are present in the dataset
    if 'text' not in df.columns or 'label' not in df.columns:
        print("Error: Required columns 'text' and 'label' not found in the dataset.")
        sys.exit(1)
    
    return df['text'], df['label']

def train_model(data_path, model_path, vectorizer_path):
    # Load the training data
    X, y = load_data(data_path)

    # Split the data into training and validation sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the TF-IDF vectorizer and fit it to the training data
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Initialize and train the Logistic Regression model
    log_reg = LogisticRegression()
    try:
        log_reg.fit(tfidf_train, y_train)
    except Exception as e:
        print(f"Error training model: {e}")
        sys.exit(1)

    # Save the trained model and vectorizer
    try:
        joblib.dump(log_reg, model_path)
        joblib.dump(tfidf_vectorizer, vectorizer_path)
        print(f"Model saved to '{model_path}'")
        print(f"Vectorizer saved to '{vectorizer_path}'")
    except Exception as e:
        print(f"Error saving model or vectorizer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up argument parser for command-line interface
    parser = argparse.ArgumentParser(description="Train a text classification model with Logistic Regression")
    # Set the default data_path to your dataset path
    parser.add_argument('--data_path', type=str, default='data/processed/processed_data.csv', help="Path to the input CSV data file")
    parser.add_argument('--model_path', type=str, default='data/models/logistic_regression_model.pkl', help="Path to save the trained model")
    parser.add_argument('--vectorizer_path', type=str, default='data/models/tfidf_vectorizer.pkl', help="Path to save the TF-IDF vectorizer")

    args = parser.parse_args()

    # Call the training function with provided arguments
    train_model(args.data_path, args.model_path, args.vectorizer_path)