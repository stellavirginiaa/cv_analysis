import streamlit as st
import re
import nltk
import PyPDF2
import joblib
import pickle
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to preprocess text
def preprocess_text(text, stopwords_list):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords_list]
    return ' '.join(tokens)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        # Save the uploaded PDF file temporarily
        with open('temp_pdf.pdf', 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Read and extract text from the saved PDF file
        with open('temp_pdf.pdf', 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
        
        # Remove the temporary file after extraction
        os.remove('temp_pdf.pdf')

        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

# Function to preprocess text from PDF
def preprocess_pdf_text(cv_text, vectorizer, stopwords_list):
    if cv_text:
        processed_cv_text = preprocess_text(cv_text, stopwords_list)
        cv_vector = vectorizer.transform([processed_cv_text])
        return cv_vector
    else:
        return None

# Load TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load model
model_path = 'logistic_regression_model.joblib'
model = joblib.load(model_path)

# Load stopwords
nltk.download('stopwords')
stopwords_list = stopwords.words('english')

# Streamlit app
def main():
    st.title('Curriculum Vitae Smart Analysis')

    """
This application predicts the category of resumes uploaded in PDF format. It leverages advanced natural language processing (NLP) techniques to extract text from PDF files, meticulously preprocess the text by eliminating stopwords and special characters, and transform it into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) representation. The processed textual data is then inputted into a logistic regression model meticulously trained to categorize resumes into various predefined fields such as "Data Science," "Web Development," "Sales," "Marketing," and many others. This seamless integration of NLP and machine learning facilitates efficient categorization and enhances the recruitment process by automating resume screening based on job-specific criteria.
"""


    # File upload for new resume
    st.subheader('Drop Your Resume Here!')
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        cv_text = extract_text_from_pdf(uploaded_file)
        if cv_text:
            cv_vector = preprocess_pdf_text(cv_text, vectorizer, stopwords_list)
            if cv_vector is not None:
                prediction = model.predict(cv_vector)
                st.markdown(f"<h2 style='text-align: center; color: #ff6347;'>Prediction</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 18px; font-weight: bold;'>Predicted Category: {prediction[0]}</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
