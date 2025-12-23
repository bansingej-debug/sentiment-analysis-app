import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# App title
st.set_page_config(page_title="Sentiment Analysis App")
st.title("💬 Sentiment Analysis App")
st.write("Analyze whether a text is Positive or Negative using Machine Learning")

# Dataset
data = {
    'text': [
        'I love this product',
        'This is the worst experience',
        'Amazing service',
        'Not good at all',
        'I am very happy',
        'I hate this',
        'This app is very helpful',
        'The app is useful',
        'Excellent and helpful application',
        'Very bad quality',
        'Poor performance',
        'Great experience'
    ],
    'sentiment': [
        'positive','negative','positive','negative','positive','negative',
        'positive','positive','positive','negative','negative','positive'
    ]
}

df = pd.DataFrame(data)

# Model training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

model = MultinomialNB()
model.fit(X, y)

# User input
user_text = st.text_area("✍️ Enter text to analyze")

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:
        text_vector = vectorizer.transform([user_text])
        prediction = model.predict(text_vector)[0]

        if prediction == "positive":
            st.success("😊 Sentiment: POSITIVE")
        else:
            st.error("☹️ Sentiment: NEGATIVE")
