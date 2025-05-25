import numpy as np
from Classification import Knn_with_OSR
from Syntetic_data import X_train, X_test, y_train, y_test

# Inicjalizacja klasyfikatora KNN z k=3
knn = Knn_with_OSR(k=3)

# Trenowanie modelu na danych treningowych
knn.fit(X_train, y_train)

# Testowanie modelu na danych testowych z różnymi metrykami
# metrics = ["euclidean", "manhattan", "minkowski", "squared_euclidean", "chebyshev"] - tu dla wszystkich metryk
metrics = ["euclidean"] # - tu dla pojedynczej metryki

print("Wyniki predykcji dla różnych metryk odległości:")
for metric in metrics:
    predictions = knn.predict(X_test, metric=metric)
    print(f"Metryka: {metric}, Predykcje: {predictions}")

# Obliczanie i wyświetlanie procentu poprawnych predykcji
print("\nProcent poprawnych predykcji dla różnych metryk odległości:")
for metric in metrics:
    predictions = knn.predict(X_test, metric=metric)
    accuracy = np.mean(predictions == y_test) * 100
    print(f"Metryka: {metric}, Procent poprawnych predykcji: {accuracy:.2f}%")