import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load CSV
df = pd.read_csv("labeled_sentiment_comments.csv")
df = df[['comment_text', 'sentiment']].dropna()

# Split
X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['sentiment'], test_size=0.2, random_state=42)

# Build full pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('nb', MultinomialNB())
])

# Train
pipeline.fit(X_train, y_train)

# Save full pipeline
joblib.dump(pipeline, "model/sentiment_model.pkl")

print("✅ Full pipeline saved to 'model/sentiment_model.pkl'")
