# Hate Speech Detection Using Naive Bayes

## Overview
This project implements a hate speech detection model using Natural Language Processing (NLP) techniques and a Naive Bayes classifier. The dataset consists of labeled tweets, and the goal is to classify whether a tweet contains hate speech or not.

## Features
- Data preprocessing (text cleaning, tokenization, TF-IDF feature extraction)
- Visualization of class distribution
- Training and evaluation of a Naive Bayes model
- Confusion matrix visualization
- Predictions on a test dataset

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset
The dataset consists of:
- `train.csv`: Contains labeled tweets with columns:
  - `tweet`: The text of the tweet
  - `label`: 0 (Non-Hate Speech) or 1 (Hate Speech)
- `test.csv`: Contains tweets without labels for predictions

## Usage
1. Load the training and test datasets:
    ```python
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    ```
2. Preprocess the text data by converting it to lowercase, removing user mentions (`@user`), and non-alphabetic characters.
3. Extract features using TF-IDF vectorization.
4. Split the data into training and validation sets.
5. Train a Naive Bayes classifier.
6. Evaluate the model using a classification report and confusion matrix.
7. Predict labels for the test dataset.

## Visualization
- A bar chart is used to display the class distribution of labels in the training set.
- A confusion matrix visualizes the model's performance on the validation set.

## Model Training
The `MultinomialNB` classifier from `sklearn.naive_bayes` is used to train the model:
```python
model = MultinomialNB()
model.fit(X_train, y_train)
```

## Evaluation
Performance is evaluated using a classification report:
```python
from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))
```

## Confusion Matrix
A heatmap of the confusion matrix is plotted using Seaborn:
```python
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
```

## Predicting on Test Data
Once the model is trained, predictions are made on the test dataset:
```python
test_df['predicted_label'] = model.predict(X_test)
```

## Results
- The model effectively classifies tweets as hate speech or non-hate speech.
- Future improvements could include more advanced preprocessing techniques, additional features, and experimenting with other machine learning models.

## License
This project is open-source and free to use.

