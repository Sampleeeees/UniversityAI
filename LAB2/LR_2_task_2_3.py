from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Input data
X = 10
y = 20

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створення класифікатора з сигмоїдальним ядром
classifier_sigmoid = SVC(kernel='sigmoid')
classifier_sigmoid.fit(X_train, y_train)

# Оцінка результатів
y_pred_sigmoid = classifier_sigmoid.predict(X_test)
accuracy_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
f1_sigmoid = cross_val_score(classifier_sigmoid, X, y, scoring='f1_weighted', cv=3).mean()

# Виведення результатів
print(f"Accuracy for Sigmoid Kernel: {accuracy_sigmoid * 100:.2f}%")
print(f"F1-Score for Sigmoid Kernel: {f1_sigmoid * 100:.2f}%")
print(classification_report(y_test, y_pred_sigmoid))
