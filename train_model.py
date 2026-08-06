import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import os
import urllib.request
import zipfile

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

def download_and_prepare_data():
    """Downloads the SMS Spam Collection dataset and saves it as spam.csv"""
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "smsspamcollection.zip"
    csv_path = "spam.csv"

    if not os.path.exists(csv_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(dataset_url, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
            
        # The extracted file is named 'SMSSpamCollection' (no extension) and is tab-separated
        extracted_file = "SMSSpamCollection"
        
        # Load and save as standard CSV
        df = pd.read_csv(extracted_file, sep='\t', header=None, names=['target', 'text'])
        df.to_csv(csv_path, index=False)
        
        # Clean up temporary files
        os.remove(zip_path)
        os.remove(extracted_file)
        os.remove("readme") # Usually included in the zip
        print("Dataset prepared and saved as spam.csv!")
    else:
        print("Dataset spam.csv already exists.")

def clean_text(text):
    """Cleans text data for machine learning"""
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

def train_and_save_model():
    """Trains the Naive Bayes model and saves it using pickle"""
    # 1. Load dataset using pandas
    print("Loading data...")
    df = pd.read_csv("spam.csv")
    
    # Convert categorical target to numeric (ham: 0, spam: 1)
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    
    # 2. Clean text data
    print("Cleaning data...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # 3. Use TF-IDF Vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['target'].values
    
    # 4. Split dataset into training and testing
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Train model using Multinomial Naive Bayes
    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # 6. Print model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # 7. Save trained model and vectorizer
    print("Saving model and vectorizer...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Training complete! Files saved: model.pkl, vectorizer.pkl")

if __name__ == "__main__":
    download_and_prepare_data()
    train_and_save_model()
