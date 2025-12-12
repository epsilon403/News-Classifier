import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
custom_stopwords = {'http', 'https', 'amp', 'co', 'new', 'get', 'like', 'would'}
stop_words.update(custom_stopwords)

svc = joblib.load('svc_model.pkl')
model = joblib.load('sentence_transformer.pkl')

label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    return " ".join(filtered_words)

def predict_text(text):
    cleaned = clean_text(text)
    emb = model.encode([cleaned])
    pred = svc.predict(emb)[0]
    return label_map[pred]

st.title("News Classifier")

text = st.text_area("Enter news text:")

if st.button("Classify"):
    if text:
        category = predict_text(text)
        st.success(f"Category: {category}")
    else:
        st.warning("Please enter some text")
