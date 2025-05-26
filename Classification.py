import numpy as np
from sklearn.base import BaseEstimator


class Knn_with_OSR(BaseEstimator):

    def __init__(self, k, threshold):
        self.k = k
        self.threshold = threshold

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_dist(self, x_test, x_train):
        return np.sqrt(np.sum((x_test - x_train) ** 2))

    def manhattan_dist(self, x_test, x_train):
        return np.sum(np.abs(x_test - x_train))

    def minkowski_dist(self, x_test, x_train, p=3):
        return np.sum(np.abs(x_test - x_train) ** p) ** (1 / p)

    def squared_euclidean_dist(self, x_test, x_train):
        return np.sum((x_test - x_train) ** 2)

    def chebyshev_dist(self, x_test, x_train):
        return np.max(np.abs(x_test - x_train))

    def predict(self, X_test, metric="euclidean"):
        # Wywolanie metryk do listy "distance_functions"
        distance_functions = {
            "euclidean": self.euclidean_dist,
            "manhattan": self.manhattan_dist,
            "minkowski": self.minkowski_dist,
            "squared_euclidean": self.squared_euclidean_dist,
            "chebyshev": self.chebyshev_dist
        }

        # Sprawdzenie czy podana metryka istnieje
        if metric not in distance_functions:
            raise ValueError(f"Unsupported metric: {metric}")

        predictions = []
        for x_test in X_test:
            # Obliczanie odległości do każdego punktu w zbiorze treningowym
            distances = [distance_functions[metric](x_test, x_train) for x_train in self.X_train]
            # Wyznaczenie k najbliższych sąsiadów
            k_indices = np.argsort(distances)[:self.k]
            # Pobranie etykiet k najbliższych sąsiadów
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # Głosowanie większościowe do której klasy przydzielić nowy punkt
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            min_distance = np.min(distances)

            # Sprawdzenie progu odległości
            if min_distance > self.threshold:
                most_common = max(set(self.y_train)) + 1
            # Dodanie klasy do listy predykcji
            predictions.append(most_common)
        # Zwrócenie tablicy z predykcjami
        return np.array(predictions)