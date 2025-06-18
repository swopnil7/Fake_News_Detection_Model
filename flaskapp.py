import os
import pytesseract
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import joblib
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load your Fake News Detection model
model_path = 'data/models/logistic_regression_model.joblib'
vectorizer_path = 'data/models/tfidf_vectorizer.joblib'
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        title = request.form['title']
        text = request.form['text']
        combined_input = title + " " + text

        transformed_input = vectorizer.transform([combined_input])
        prediction = model.predict(transformed_input)[0]
        confidence_score = model.predict_proba(transformed_input).max() * 100

        label = "Real News" if prediction == 1 else "Fake News"
        result = f"{label} ({confidence_score:.2f}% Confidence)"

        return render_template('home.html', prediction=result, active_tab="text")
    except Exception as e:
        return render_template('home.html', prediction=f"Error: {str(e)}", active_tab="text")


@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return render_template('home.html', ocr_error="No file part", active_tab="image")

    file = request.files['image']
    if file.filename == '':
        return render_template('home.html', ocr_error="No selected file", active_tab="image")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        try:
            image = cv2.imread(image_path)
            extracted_text = pytesseract.image_to_string(image)

            transformed_input = vectorizer.transform([extracted_text])
            prediction = model.predict(transformed_input)[0]
            confidence_score = model.predict_proba(transformed_input).max() * 100

            label = "Real News" if prediction == 1 else "Fake News"
            result = f"{label} ({confidence_score:.2f}% Confidence)"

            return render_template('home.html', extracted_text=extracted_text, prediction=result, active_tab="image")
        except Exception as e:
            return render_template('home.html', ocr_error=f"Error during OCR processing: {str(e)}", active_tab="image")
    else:
        return render_template('home.html', ocr_error="Unsupported file type", active_tab="image")


if __name__ == '__main__':
    app.run(debug=True)
