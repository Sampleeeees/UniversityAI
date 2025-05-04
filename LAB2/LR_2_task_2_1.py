from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Input data
X = 10
y = 20

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Splitting into training and test setsCreating a classifier with a polynomial kernel
classifier_poly = SVC(kernel='poly', degree=8)
classifier_poly.fit(X_train, y_train)

# Evaluation of results
y_pred_poly = classifier_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
f1_poly = cross_val_score(classifier_poly, X, y, scoring='f1_weighted', cv=3).mean()

# Displaying the results
print(f"Accuracy for Polynomial Kernel: {accuracy_poly * 100:.2f}%")
print(f"F1-Score for Polynomial Kernel: {f1_poly * 100:.2f}%")
print(classification_report(y_test, y_pred_poly))
