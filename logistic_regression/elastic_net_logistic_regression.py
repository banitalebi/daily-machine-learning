import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# The Breast Cancer dataset
cancer_dataset = datasets.load_breast_cancer()
X = cancer_dataset.data
y = cancer_dataset.target

cancer_df = pd.DataFrame(data=X, columns=cancer_dataset.feature_names)
cancer_df['target'] = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Here we use the 'penalty' parameter to specify 'elasticnet' and set l1_ratio for the mix of L1 and L2 regularization.
logreg = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000)

# Train the model
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Probability of the positive class
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Output the results
print(f'Accuracy of Elastic Net logistic regression classifier: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)
print(f'ROC AUC Score: {roc_auc:.2f}')

# Predicting the probability of a new sample
new_sample = np.array([[13.0, 15.0, 85.0, 550.0,
                        0.1, 0.06, 0.02,
                        0.06, 0.08, 0.15,
                        0.4, 1.2, 3.1,
                        12.5, 15.5, 90.0,
                        550.0, 0.05, 0.05,
                        0.01, 0.06, 0.01,
                        0.05, 0.07, 0.15,
                        1.4, 3.2, 12.5,
                        15.5, 90.0]])
predicted_prob = logreg.predict_proba(new_sample)
print(f'Predicted probability of being malignant: {predicted_prob[0][1]:.2f}')
