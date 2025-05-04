from sklearn.datasets import load_iris
import pandas as pd


iris_dataset = load_iris()
print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))
print("Назви відповідей: {}".format(iris_dataset['target_names']))
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))


# Перетворюємо дані в DataFrame для зручності
iris_data = pd.DataFrame(data=iris_dataset['data'], columns=iris_dataset['feature_names'])
iris_data['target'] = iris_dataset['target']

print(iris_data.head(5))


print(iris_data.describe())


print(iris_data.groupby('target').size())


import matplotlib.pyplot as plt

iris_data.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
plt.show()


from pandas.plotting import scatter_matrix

scatter_matrix(iris_data, figsize=(10,10))
plt.show()


from sklearn.model_selection import train_test_split

X = iris_dataset['data']
y = iris_dataset['target']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

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
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Оцінка якості моделі
print("Accuracy: ", accuracy_score(Y_validation, predictions))
print("Confusion Matrix: \n", confusion_matrix(Y_validation, predictions))
print("Classification Report: \n", classification_report(Y_validation, predictions))


import numpy as np

# Нові дані
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = model.predict(X_new)

# Виведення результату
print("Predicted class: ", iris_dataset['target_names'][prediction])
