import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("news.csv", engine="python", on_bad_lines="skip")

# Remove missing values
df.dropna(inplace=True)

# ===========================
# Features & Target
# ===========================

X = df["text"]
y = df["label"]

# ===========================
# Train-Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===========================
# TF-IDF Vectorizer
# ===========================

vectorizer = TfidfVectorizer(stop_words="english")

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# ===========================
# Train Logistic Regression Model
# ===========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vectorized, y_train)

# ===========================
# Prediction
# ===========================

y_pred = model.predict(X_test_vectorized)

# ===========================
# Evaluation
# ===========================

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Model Accuracy :", round(accuracy * 100, 2), "%")
print("=" * 50)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===========================
# Save Model & Vectorizer
# ===========================

joblib.dump(model,"fake_news_model.pkl")
joblib.dump(vectorizer, vectorizer.pkl")

print("Model and vectorizer saved successfully")

