import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load data
train = pd.read_csv('banking77_train.csv')
test = pd.read_csv('banking77_test.csv')

print(f"Training samples: {len(train)}")
print(f"Testing samples: {len(test)}")

# Features and labels
X_train = train['text']
y_train = train['category']
X_test = test['text']
y_test = test['category']

# TF-IDF Vectorizer
print("\nConverting text to numbers using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

# Train model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tf, y_train)

# Evaluate
preds = model.predict(X_test_tf)
accuracy = accuracy_score(y_test, preds)
print(f"\n✅ Accuracy: {accuracy:.2%}")

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
print("\n✅ Model saved as model.pkl")
print("✅ Vectorizer saved as tfidf.pkl")