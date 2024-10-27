import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request, jsonify
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

df = pd.read_csv('Tweets.csv')
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000) 
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['text'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['text'])

y_train = train_data['airline_sentiment']
y_test = test_data['airline_sentiment']

model = LogisticRegression(max_iter=1000)  
model.fit(X_train_tfidf, y_train)

def predict_sentiment(text):
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
    img = io.BytesIO()
    FigureCanvas(plt.gcf()).print_png(img)
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    choice = request.form['choice']
    extra_input = request.form.get('extra_input', '')
    
    if choice == '1':
        data = df
        category = 'Overall'
    elif choice == '2':
        data = df[df['user_timezone'] == extra_input]
        category = extra_input + ' Timezone'
    elif choice == '3':
        data = df[df['tweet_location'] == extra_input]
        category = extra_input + ' Location'
    elif choice == '4':
        data = df[df['airline'] == extra_input]
        category = extra_input + ' Airline'
    elif choice == '5':
        predicted_sentiment = predict_sentiment(extra_input)
        return jsonify({'sentiment': predicted_sentiment})
    
    positive_percentage = calculate_percentage(data, 'positive')
    negative_percentage = calculate_percentage(data, 'negative')
    neutral_percentage = calculate_percentage(data, 'neutral')
    plot_url = plot_sentiment_distribution(data, category)
    
    return jsonify({
        'positive_percentage': positive_percentage,
        'negative_percentage': negative_percentage,
        'neutral_percentage': neutral_percentage,
        'plot_url': plot_url
    })

if __name__ == '__main__':
    app.run(debug=True)
