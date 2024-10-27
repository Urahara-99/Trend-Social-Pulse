import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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
    plt.show()

while True:
    print("Options:")
    print("1. Overall Sentiment Percentages")
    print("2. Sentiment Percentages by User Timezone")
    print("3. Sentiment Percentages by Tweet Location")
    print("4. Sentiment Percentages by Airline")
    print("5. Predict Sentiment for a Text")
    print("6. Exit")

    user_input = input("Enter your choice (1-6): ")

    if user_input == '1':
        overall_positive_percentage = calculate_percentage(df, 'positive')
        overall_negative_percentage = calculate_percentage(df, 'negative')
        overall_neutral_percentage = calculate_percentage(df, 'neutral')

        print(f"Overall Positive Tweets Percentage: {overall_positive_percentage:.2f}%")
        print(f"Overall Negative Tweets Percentage: {overall_negative_percentage:.2f}%")
        print(f"Overall Neutral Tweets Percentage: {overall_neutral_percentage:.2f}%")

        plot_sentiment_distribution(df, 'Overall')

    elif user_input == '2':
        user_timezone = input("Enter User Timezone: ")
        timezone_data = df[df['user_timezone'] == user_timezone]
        if len(timezone_data) > 0:
            timezone_positive_percentage = calculate_percentage(timezone_data, 'positive')
            timezone_negative_percentage = calculate_percentage(timezone_data, 'negative')
            timezone_neutral_percentage = calculate_percentage(timezone_data, 'neutral')

            print(f"Positive Tweets Percentage for {user_timezone}: {timezone_positive_percentage:.2f}%")
            print(f"Negative Tweets Percentage for {user_timezone}: {timezone_negative_percentage:.2f}%")
            print(f"Neutral Tweets Percentage for {user_timezone}: {timezone_neutral_percentage:.2f}%")

            plot_sentiment_distribution(timezone_data, f'{user_timezone} Timezone')

        else:
            print(f"No data available for {user_timezone}. Please try another timezone.")

    elif user_input == '3':
        tweet_location = input("Enter Tweet Location: ")
        location_data = df[df['tweet_location'] == tweet_location]
        if len(location_data) > 0:
            location_positive_percentage = calculate_percentage(location_data, 'positive')
            location_negative_percentage = calculate_percentage(location_data, 'negative')
            location_neutral_percentage = calculate_percentage(location_data, 'neutral')

            print(f"Positive Tweets Percentage for {tweet_location}: {location_positive_percentage:.2f}%")
            print(f"Negative Tweets Percentage for {tweet_location}: {location_negative_percentage:.2f}%")
            print(f"Neutral Tweets Percentage for {tweet_location}: {location_neutral_percentage:.2f}%")

            plot_sentiment_distribution(location_data, f'{tweet_location} Location')

        else:
            print(f"No data available for {tweet_location}. Please try another location.")

    elif user_input == '4':
        airline = input("Enter Airline: ")
        airline_data = df[df['airline'] == airline]
        if len(airline_data) > 0:
            airline_positive_percentage = calculate_percentage(airline_data, 'positive')
            airline_negative_percentage = calculate_percentage(airline_data, 'negative')
            airline_neutral_percentage = calculate_percentage(airline_data, 'neutral')

            print(f"Positive Tweets Percentage for {airline}: {airline_positive_percentage:.2f}%")
            print(f"Negative Tweets Percentage for {airline}: {airline_negative_percentage:.2f}%")
            print(f"Neutral Tweets Percentage for {airline}: {airline_neutral_percentage:.2f}%")

            plot_sentiment_distribution(airline_data, f'{airline} Airline')

        else:
            print(f"No data available for {airline}. Please try another airline.")

    elif user_input == '5':
        text_input = input("Enter a text for sentiment prediction: ")
        predicted_sentiment = predict_sentiment(text_input)
        print(f"Predicted Sentiment: {predicted_sentiment}")

    elif user_input == '6':
        break

    else:
        print("Invalid choice. Please enter a number between 1 and 6.")
