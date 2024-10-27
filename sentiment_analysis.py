import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def initialize_model(df):
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['text'])
    y_train = train_data['airline_sentiment']
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    return model, tfidf_vectorizer, train_data

def predict_sentiment(text, model, tfidf_vectorizer):
    text_tfidf = tfidf_vectorizer.transform([text])
    predicted_sentiment = model.predict(text_tfidf)[0]
    return predicted_sentiment

def calculate_percentage(data, sentiment):
    sentiment_percentage = (len(data[data['airline_sentiment'] == sentiment]) / len(data)) * 100
    return sentiment_percentage

def plot_sentiment_distribution(data, category):
    sentiment_counts = data['airline_sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'yellow', 'red'])
    plt.title(f'Sentiment Distribution for {category}')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
