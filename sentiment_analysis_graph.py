import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

plt.style.use('ggplot')

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Display basic info
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(train_df.head())

# Check class distribution
ax = train_df['label'].value_counts().plot(kind='bar', 
                                           title='Distribution of Hate Speech Labels',
                                           figsize=(10, 5))
ax.set_xlabel('Label (0: Non-Hate, 1: Hate Speech)')
ax.set_ylabel('Count')
plt.show()

# Preprocess text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.replace('@user', '')  # Remove user mentions
    text = text.replace(r'[^a-zA-Z\s]', '')  # Remove non-alphabetic characters
    return text

train_df['tweet'] = train_df['tweet'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train_df['tweet'])
y = train_df['label']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Test dataset prediction
test_df['tweet'] = test_df['tweet'].apply(preprocess_text)
X_test = vectorizer.transform(test_df['tweet'])
test_df['predicted_label'] = model.predict(X_test)
