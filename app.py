import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

def clean_text(text):
    """Cleans text data exactly as it was done during training"""
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

# Set page configuration for centered layout
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="centered")

# Load saved model and vectorizer
@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please run train_model.py first.")
        return None, None

model, vectorizer = load_models()

# Add modern UI styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    h1 {
        color: #1f2937;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #d1d5db;
        padding: 15px;
        font-size: 16px;
    }
    .stButton button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #4338ca;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        animation: fadeIn 0.5s;
    }
    .spam {
        background-color: #fee2e2;
        color: #dc2626;
        border: 2px solid #f87171;
    }
    .ham {
        background-color: #dcfce7;
        color: #16a34a;
        border: 2px solid #4ade80;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Add title
st.title("Spam Email Classifier 📧")
st.markdown("<p style='text-align: center; color: #4b5563; margin-bottom: 30px;'>Enter an email or SMS message below to check if it's Spam or Ham.</p>", unsafe_allow_html=True)

# Add text area for user input
user_input = st.text_area("Message Content:", height=150, placeholder="Paste your message here...")

# Add Predict button
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to predict.")
    elif model is not None and vectorizer is not None:
        # Preprocess input
        cleaned_text = clean_text(user_input)
        
        # Transform user input using vectorizer
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()
        
        # Predict spam or ham
        prediction = model.predict(vectorized_text)[0]
        
        # Display result with proper styling (Spam in red, Ham in green)
        if prediction == 1:
            st.markdown('<div class="result-box spam">🚨 This message is SPAM</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box ham">✅ This message is NOT SPAM (Ham)</div>', unsafe_allow_html=True)

# Footer instructions
st.markdown("---")
st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 14px;'>Built with Python, Scikit-learn, and Streamlit</p>", unsafe_allow_html=True)
