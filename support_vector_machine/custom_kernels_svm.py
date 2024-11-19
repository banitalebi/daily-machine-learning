import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Load the Wine dataset
wine_dataset = datasets.load_wine()
X = wine_dataset.data
y = wine_dataset.target

# Create a DataFrame for visualization (Optional)
wine_df = pd.DataFrame(data=X, columns=wine_dataset.feature_names)
wine_df['target'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define a custom polynomial kernel function
def polynomial_kernel(X1, X2):
    return (np.dot(X1, X2.T) + 1) ** 2  # Polynomial kernel of degree 2

# Define a more complex polynomial kernel function
def complex_polynomial_kernel(X1, X2):
    gamma = 1 / X1.shape[1]  # Example scaling factor
    r = 1  # Bias term
    d = 4  # Degree of the polynomial
    return (gamma * np.dot(X1, X2.T) + r) ** d

# Initialize the SVM with the custom kernel wrapped in OneVsRestClassifier
svm_model = OneVsRestClassifier(SVC(kernel=polynomial_kernel, probability=True))

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Get the probability of each class for the test set
y_pred_proba = svm_model.predict_proba(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# Display results
print(f'Accuracy of Polynomial SVM classifier (One-vs-Rest): {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)
print(f'ROC AUC Score: {roc_auc:.2f}')

# Predicting the probability of a new sample
new_sample = np.array([[13.0, 1.5, 2.5, 20.0,
                        90.0, 0.5, 1.5,
                        0.5, 0.3, 1.0,
                        2.0, 3.0, 750.0]])

predicted_prob = svm_model.predict_proba(new_sample)
print(f'Predicted probabilities for new sample: {predicted_prob}')