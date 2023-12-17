import streamlit as st 
import pickle
from nltk.corpus import stopwords
import nltk 
import string 
from nltk.stem.porter import PorterStemmer
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]       
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Assuming you have a trained model and vectorizer
tfidf = TfidfVectorizer()  # Replace with your actual vectorizer
model = None  # Replace with your actual model

# Save the vectorizer and model
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

st.title('Email Spam Classifier')

input_sms = st.text_input('Enter the Message ')

option = st.selectbox("You Got Message From :-", ["Via Email ", "Via SMS", "other"])

if st.checkbox("Check me"):
    st.write("")

if st.button('Click to Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')
