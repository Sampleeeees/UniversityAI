import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from typing import List

# Input file containing the data
input_file: str = 'income_data.txt'

# Reading the data from the file
X: List[List[str]] = []  # Features
y: List[int] = []        # Target labels
count_class1: int = 0    # Counter for class <=50K
count_class2: int = 0    # Counter for class >50K
max_datapoints: int = 25000  # Maximum number of datapoints for each class

# Open and process the file
with open(input_file, 'r') as f:
    for line in f.readlines():
        # Stop when we reach the maximum number of datapoints for both classes
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        # Skip lines with missing data
        if '?' in line:
            continue

        # Split the data line by comma
        data: List[str] = line[:-1].split(", ")

        # Assign data to the respective class (<=50K or >50K)
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])  # Exclude the target label from features
            y.append(0)  # Target label for <=50K
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])  # Exclude the target label from features
            y.append(1)  # Target label for >50K
            count_class2 += 1

# Convert features into a NumPy array
X = np.array(X)

# Convert string data to numerical values using label encoding
label_encoder = []  # To store label encoders for categorical columns
X_encoded = np.empty(X.shape)  # To store the encoded data

# Encoding the features (excluding the target variable)
for i, item in enumerate(X[0]):
    if item.isdigit():
        # If the item is already numeric, leave it as is
        X_encoded[:, i] = X[:, i]
    else:
        # Encode categorical data
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

# Convert features and target labels to integer values
X = X_encoded[:, :-1].astype(int)  # Features (excluding the last column)
y = np.array(y)  # Target labels (last column)

# Check the shape of X and y before proceeding
print("X shape:", X.shape)
print("y shape:", len(y))

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Create and train the SVM classifier using OneVsOne strategy
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
classifier.fit(X_train, y_train)

# Predict the results on the test set
y_test_pred = classifier.predict(X_test)

# Calculate the F1-score using cross-validation
f1_scores = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print(f"F1 score: {round(100 * f1_scores.mean(), 2)}%")

# Calculate other classification metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

# Print the metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 score (weighted): {f1 * 100:.2f}%")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:\n", cm)

# Example input data for prediction (needs encoding)
input_data: List[str] = [
    '37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
    'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States'
]

# Encode the test data using the same encoders as the training data
input_data_encoded: List[int] = [-1] * len(input_data)
count: int = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])  # Access the first element
        count += 1

# Convert the encoded input data to a NumPy array
input_data_encoded = np.array(input_data_encoded)

input_data_encoded = input_data_encoded[:13]

# Use the trained classifier to predict the class for the new data point
predicted_class = classifier.predict(input_data_encoded.reshape(1, -1))

# Inverse transform to get the original label
predicted_label = label_encoder[-1].inverse_transform(predicted_class)[0]
print(f"Predicted class for the input data: {predicted_label}")
