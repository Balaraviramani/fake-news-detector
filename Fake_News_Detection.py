# Fake News Detection - Python Script Version

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

nltk.download('stopwords')

# Load dataset from local files
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add labels and merge
true_df['label'] = 0  # REAL
fake_df['label'] = 1  # FAKE
df = pd.concat([true_df, fake_df]).reset_index(drop=True)


# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# Apply cleaning
df['text'] = df['text'].apply(clean_text)

# Feature and Label
X = df['text']
y = df['label'].map({'REAL': 0, 'FAKE': 1})

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'")
