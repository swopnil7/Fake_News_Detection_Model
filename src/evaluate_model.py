# evaluate_model.py
import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

def evaluate_model(data_path, model_path, vectorizer_path):
    # Load the test data
    X_test, y_test = load_data(data_path)

    try:
        # Load the trained model and vectorizer from disk
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        sys.exit(1)

    # Transform the test data using the loaded vectorizer
    X_test_transformed = vectorizer.transform(X_test)

    # Make predictions using the trained model
    y_pred = model.predict(X_test_transformed)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Generate a classification report with meaningful labels
    class_report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])

    # Output the evaluation metrics
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Format and output the confusion matrix with labels
    labels = ['Fake', 'Real']
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print(f'Confusion Matrix:\n{conf_matrix_df}')
    
    print(f'Classification Report:\n{class_report}')

if __name__ == "__main__":
    # Set up argument parser for command-line interface
    parser = argparse.ArgumentParser(description="Evaluate a trained text classification model")
    # Set default paths for data, model, and vectorizer
    parser.add_argument('--data_path', type=str, default='data/processed/processed_data.csv', help="Path to the input CSV test data file")
    parser.add_argument('--model_path', type=str, default='data/models/logistic_regression_model.joblib', help="Path to the trained model file")
    parser.add_argument('--vectorizer_path', type=str, default='data/models/tfidf_vectorizer.joblib', help="Path to the saved TF-IDF vectorizer file")

    args = parser.parse_args()

    # Call the evaluation function with provided arguments
    evaluate_model(args.data_path, args.model_path, args.vectorizer_path)