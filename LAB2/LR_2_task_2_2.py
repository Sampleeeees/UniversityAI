from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Input data
X = 10
y = 20

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створення класифікатора з гаусовим ядром
classifier_rbf = SVC(kernel='rbf')
classifier_rbf.fit(X_train, y_train)

# Оцінка результатів
y_pred_rbf = classifier_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
f1_rbf = cross_val_score(classifier_rbf, X, y, scoring='f1_weighted', cv=3).mean()

# Виведення результатів
print(f"Accuracy for RBF Kernel: {accuracy_rbf * 100:.2f}%")
print(f"F1-Score for RBF Kernel: {f1_rbf * 100:.2f}%")
print(classification_report(y_test, y_pred_rbf))
