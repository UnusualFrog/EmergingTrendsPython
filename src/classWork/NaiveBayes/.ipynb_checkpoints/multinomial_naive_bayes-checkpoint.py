# Importing Required Libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Dataset
categories = ['sci.space', 'rec.sport.hockey']  # Two categories for binary classification
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Display sample data
print(f"Sample text:\n{newsgroups.data[0]}\n")
print(f"Label: {newsgroups.target_names[newsgroups.target[0]]}")

# 2. Preprocessing: Convert Text to Numerical Data
vectorizer = CountVectorizer()  # Convert text to token counts
X = vectorizer.fit_transform(newsgroups.data)  # Feature matrix
y = newsgroups.target  # Labels

# 3. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Multinomial Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# 5. Make Predictions
y_pred = nb_classifier.predict(X_test)

# 6. Evaluate the Model
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))