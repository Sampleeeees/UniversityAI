import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO


# Завантажуємо набір даних Iris
iris = load_iris()
X, y = iris.data, iris.target

# Розбиття на навчальну та тестову вибірки (70% навчання, 30% тестування)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)


# Створення лінійного класифікатора Ridge
clf = RidgeClassifier(tol=1e-2, solver="sag")

# Навчання класифікатора на навчальній вибірці
clf.fit(Xtrain, ytrain)

# Прогнозування на тестовій вибірці
ypred = clf.predict(Xtest)



# Оцінка якості класифікації
print('Accuracy:', np.round(metrics.accuracy_score(ytest, ypred), 4))
print('Precision:', np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4))
print('Recall:', np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(ytest, ypred), 4))
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(ytest, ypred), 4))
print('\t\tClassification Report:\n', metrics.classification_report(ytest, ypred))


# Матриця плутанини
mat = confusion_matrix(ytest, ypred)

# Візуалізація матриці плутанини за допомогою seaborn
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.savefig("Confusion.jpg")

