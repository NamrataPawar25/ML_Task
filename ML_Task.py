import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def load_data(spam_dir, ham_dir):
    emails = []
    labels = []

    for filename in os.listdir(spam_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(spam_dir, filename), 'r', encoding='utf-8') as f:
                emails.append(f.read())
                labels.append('spam')

    for filename in os.listdir(ham_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(ham_dir, filename), 'r', encoding='utf-8') as f:
                emails.append(f.read())
                labels.append('ham')

    return pd.DataFrame({'email': emails, 'label': labels})

spam_dir = 'spamassassin-corpus/spam'
ham_dir = 'spamassassin-corpus/ham'

data = load_data(spam_dir, ham_dir)

X_train, X_test, y_train, y_test = train_test_split(data['email'], data['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
