import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
df = pd.read_csv("raw_data.csv", encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ["sentiment", "text"]

# Convert sentiment (0=neg, 4=pos → 0/1)
df["sentiment"] = df["sentiment"].replace(4, 1)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

# Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["clean_text"])
y = df["sentiment"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model + vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")