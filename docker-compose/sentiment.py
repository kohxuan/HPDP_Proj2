import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load your combined YouTube comments
df = pd.read_csv("Combined_HariRayaComments.csv")

# Only keep required column
df = df[['comment_text']].dropna()

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Function to classify based on compound score
def classify_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply the function to label each comment
df['sentiment'] = df['comment_text'].apply(classify_sentiment)

# Save labeled data
df.to_csv("labeled_sentiment_comments.csv", index=False)

print("✅ Auto-labeled sentiments and saved to 'labeled_sentiment_comments.csv'")
