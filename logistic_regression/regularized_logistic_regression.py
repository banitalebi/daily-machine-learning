import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Load Wine Quality dataset (using pandas to read CSV)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_df = pd.read_csv(url, sep=';')

# Define features and target
X = wine_df.drop('quality', axis=1)
y = wine_df['quality'].apply(lambda x: 1 if x >= 6 else 0)  # Binary classification: good (1) or bad (0)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a Regularized Logistic Regression model (L2 regularization)
logreg = LogisticRegression(max_iter=10000, C=1.0)  # C is the inverse of regularization strength

# Train the model
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

# Evaluate performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Display results
print(f'Accuracy of logistic regression classifier: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)
print(f'ROC AUC Score: {roc_auc:.2f}')

# Predicting the probability of a new sample
new_sample = np.array([[7.4, 0.7, 0.00, 1.9, 0.076, 11.0, 34.0, 0.99800, 3.16, 0.58, 9.8]])
predicted_prob = logreg.predict_proba(new_sample)
print(f'Predicted probability of being good quality: {predicted_prob[0][1]:.2f}')
