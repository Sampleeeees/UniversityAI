import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# Вхідний файл
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data: List[str] = line[:-1].split(", ")

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])  # Exclude the target label
            y.append(0)  # Target label for <=50K
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])  # Exclude the target label
            y.append(1)  # Target label for >50K
            count_class2 += 1

X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)  # Ознаки (всі крім останнього стовпця)
y = np.array(y)  # Мітки


# Список моделей
models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

# Оцінка моделей
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

# Візуалізація результатів
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()



# Використовуємо найкращу модель, наприклад, SVM
best_model = SVC(gamma='auto')
best_model.fit(X, y)

# Передбачення на нових даних
predictions = best_model.predict(X)

# Оцінка якості
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y, predictions))
print("Confusion Matrix:\n", confusion_matrix(y, predictions))
print("Classification Report:\n", classification_report(y, predictions))
