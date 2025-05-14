import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# Вихідні дані для завдання 2 (метод найменших квадратів)
X = np.array([1, 6, 11, 16, 21, 26])
Y = np.array([19, 37, 49, 61, 68, 90])

# Метод найменших квадратів
X_mean = np.mean(X)
Y_mean = np.mean(Y)

numerator = np.sum((X - X_mean) * (Y - Y_mean))
denominator = np.sum((X - X_mean) ** 2)
beta1 = numerator / denominator
beta0 = Y_mean - beta1 * X_mean

Y_pred = beta0 + beta1 * X

# Виведення коефіцієнтів
print(f"Метод найменших квадратів - Рівняння: y = {beta0:.4f} + {beta1:.4f}x")

# Побудова графіка методу найменших квадратів
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Експериментальні дані')
plt.plot(X, Y_pred, color='red', label=f'Апроксимація: y = {beta0:.2f} + {beta1:.2f}x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Метод найменших квадратів (Варіант 10)')
plt.grid(alpha=0.5)
plt.legend()
plt.savefig('test.png')

# Вихідні дані для інтерполяції (5 точок з завдання)
X_interp = np.array([1, 6, 11, 16, 21])
Y_interp = np.array([19, 37, 49, 61, 68])

# Інтерполяція поліномом ступеня 4
polynomial = Polynomial.fit(X_interp, Y_interp, 4)

# Генеруємо значення для графіка
X_vals = np.linspace(min(X_interp), max(X_interp), 500)
Y_vals = polynomial(X_vals)

# Обчислення значень у точках 0.2 та 0.5
y_02 = polynomial(0.2)
y_05 = polynomial(0.5)

print(f"Інтерполяція - Значення полінома у точці 0.2: {y_02:.4f}")
print(f"Інтерполяція - Значення полінома у точці 0.5: {y_05:.4f}")

# Побудова графіка інтерполяції
plt.figure(figsize=(10, 6))
plt.scatter(X_interp, Y_interp, color='blue', label='Інтерполяційні точки')
plt.plot(X_vals, Y_vals, color='red', label='Інтерполяційний поліном')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Інтерполяція функції поліномом ступеня 4 (Варіант 10)')
plt.grid(alpha=0.5)
plt.legend()
plt.savefig('output.png')