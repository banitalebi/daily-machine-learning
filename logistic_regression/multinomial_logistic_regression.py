import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# The Iris Dataset
iris = load_iris()
# Features (sepal length, sepal width, petal length, petal width)
X = iris.data
# Target variable (species)
y = iris.target  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The Multinomial Logistic Regression Model
multi_logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
multi_logreg.fit(X_train, y_train)

# Predictions
y_pred = multi_logreg.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# The results
print(f'Accuracy of logistic regression classifier: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)

# Predicting the probability of a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example of a new iris flower measurement
predicted_prob = multi_logreg.predict_proba(new_sample)
print(f'Predicted probabilities for the new sample: {predicted_prob[0]}')
print(f'Predicted probability of being species 0: {predicted_prob[0][0]:.2f}')
print(f'Predicted probability of being species 1: {predicted_prob[0][1]:.2f}')
print(f'Predicted probability of being species 2: {predicted_prob[0][2]:.2f}')
