from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords
nltk.download("stopwords", quiet=True)

app = Flask(__name__)

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    model = None
    vectorizer = None
    print("Error loading model:", e)


# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    stop_words = set(stopwords.words("english"))

    text = " ".join(
        word for word in text.split()
        if word not in stop_words
    )

    return text


# -----------------------------
# Home Page
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        message = request.form.get("message")

        if message and model and vectorizer:

            cleaned = clean_text(message)

            vector = vectorizer.transform([cleaned])

            pred = model.predict(vector)[0]

            if pred == 1:
                prediction = "🚨 This message is SPAM"
            else:
                prediction = "✅ This message is NOT SPAM (Ham)"

    return render_template(
        "index.html",
        prediction=prediction
    )


# -----------------------------
# Health Check
# -----------------------------
@app.route("/health")
def health():
    return {
        "status": "ok"
    }


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)