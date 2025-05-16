import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Sample data
data = pd.DataFrame({
    'content_type': ['Reel', 'Photo', 'Story', 'Reel', 'Photo'],
    'post_type': ['Tutorial', 'Showcase', 'Behind-the-Scenes', 'Q&A', 'Tutorial'],
    'label': ['Improve visuals', 'Use better lighting', 'Add music', 'Try humor', 'Simplify message']
})

# Combine input features
data['input'] = data['content_type'] + " " + data['post_type']

# Vectorize text inputs
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['input'])

# Encode textual labels
encoder = LabelEncoder()
y = encoder.fit_transform(data['label'])

# Train classifier
model = LogisticRegression()
model.fit(X, y)

# Save models
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/suggestion_model.pkl")
joblib.dump(vectorizer, "model/suggestion_vectorizer.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")

print("✅ Model, vectorizer, and encoder saved.")
