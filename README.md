## **Project Description**

### **Fake News Detection Model: Ensuring Trust in News**

In an era dominated by the rapid spread of information, distinguishing between genuine and fake news has become a critical challenge. This project leverages cutting-edge machine learning algorithms and text processing techniques to tackle this issue effectively.  

**Fake News Detection App** is an intuitive tool designed to classify news articles as either **Real** or **Fake** with high accuracy. It empowers users to validate the authenticity of the information they consume, promoting awareness and combating misinformation.

---

### **Features**

- **Text-Based Classification:**  
  Users can paste the text of a news article into the app, which is then analyzed and classified in real-time.

- **OCR Integration for Images:**  
  Upload an image of a news article or screenshot, and the app extracts text using Optical Character Recognition (OCR) and performs classification.

- **Interactive User Interface:**  
  The app provides a clean, responsive, and user-friendly interface powered by **Streamlit**, ensuring ease of use for all users.

- **Confidence Levels:**  
  The app not only provides a classification (Real or Fake) but also displays the confidence level of its prediction.

---

### **Technologies and Tools**

- **Machine Learning Algorithms:**  
  A trained Logistic Regression model for high-accuracy text classification.
  
- **NLP Techniques:**  
  Utilizes TF-IDF vectorization for feature extraction from text data.

- **Optical Character Recognition (OCR):**  
  Integrated with **pytesseract** to extract text from uploaded images seamlessly.

- **Streamlit Framework:**  
  Enables the creation of a visually appealing and interactive web app.

- **Joblib for Model Serialization:**  
  Ensures efficient storage and loading of the trained model and vectorizer.
  
---

### [How to run this project]
- Step 1: Clone the repository in your device:
>   git clone [<repository_url>](https://github.com/Nepal-College-of-Information-Technology/the-project-work-keyboard_crackers.git)
>>   cd <repository_directory>

- Step 2: Set up a virtual python environment:
>   python -m venv env
>>   source .env\Scripts\activate

- Step 3: Install the required Python libraries:
>   pip install -r requirements.txt

- Step 4: Run the Flask/Streamlit application:
>   Streamlit: python app.py or, [click here](https://fakenewsdetectionmodel.streamlit.app/)
>>   Flask: python flaskapp.py