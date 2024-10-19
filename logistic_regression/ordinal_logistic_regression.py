import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# The Wine Quality dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')
print(df.head())

# Categorizing the quality scores into ordinal classes
def categorize_quality(quality):
    if quality <= 4:
        return 'Low'
    elif quality <= 6:
        return 'Medium'
    else:
        return 'High'

df['quality'] = df['quality'].apply(categorize_quality)
df['quality'] = pd.Categorical(df['quality'], ordered=True)

# Splitting the dataset into features and target variable
X = df.drop(columns=['quality'])
# Converting categories to codes for modeling
y = df['quality'].cat.codes  

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Fitting the Ordinal Logistic Regression model
model = OrderedModel(y_train, X_train, distr='logit')
res = model.fit(method='bfgs')

# Summary of the model
print(res.summary())

# Making predictions on the test set
y_pred_probs = res.predict(X_test)
# Get the class with the highest probability
y_pred = np.argmax(y_pred_probs, axis=1)

# Evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# The results
print(f'Accuracy of ordinal logistic regression classifier: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)

# Predicting the probability of a new sample
new_sample = np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.16, 0.58]])
new_sample_probs = res.predict(new_sample)
print(f'Predicted probabilities for new sample: {new_sample_probs}')
