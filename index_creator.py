import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy import sparse
from nltk.tokenize import word_tokenize
import requests


def custom_preprocessor(text):
    data = {"text": text}
    response = requests.post("http://127.0.0.1:8001/text-processing", json=data)
    return response.json().get('processed_text')


def get_documents(dataset_path):
    df = pd.read_csv(dataset_path, usecols=[0, 1])    
    content_column_name = df.columns[1]
    documents = list(df[content_column_name])
    return documents


def save_index(vectorizer, tfidf_matrix):    
    joblib.dump(vectorizer, "vectorizer.pkl")
    sparse.save_npz("tfidf_matrix.npz", tfidf_matrix)


def create_index(dataset_path):
    documents = get_documents(dataset_path)    
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, tokenizer=word_tokenize, preprocessor=custom_preprocessor, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(documents)
    save_index(vectorizer, tfidf_matrix)