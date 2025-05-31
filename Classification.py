import numpy as np
from sklearn.base import BaseEstimator

class Knn_with_OSR(BaseEstimator):

    def __init__(self, k, threshold=None, threshold_percentile=95, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.threshold = threshold
        self.threshold_percentile = threshold_percentile

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        # Wywolanie metryk do listy "distance_functions"
        distance_functions = {
            "euclidean": self.euclidean_dist,
            "manhattan": self.manhattan_dist,
            "minkowski": self.minkowski_dist,
            "squared_euclidean": self.squared_euclidean_dist,
            "chebyshev": self.chebyshev_dist
        }
        # Wyznacz dystanse między każdą próbką treningową a innymi
        if self.threshold is None:
            distances = []
            for i, x in enumerate(X_train):
                # pomijamy siebie
                other_X = np.delete(X_train, i, axis=0)
                dists = [distance_functions[self.metric](x, x_other) for x_other in other_X]
                distances.append(np.min(dists))  # można też użyć średniej z k
            # Ustaw próg jako percentyl (np. 95)
            self.threshold = np.percentile(distances, self.threshold_percentile)

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

    def predict(self, X_test):
        # Wywolanie metryk do listy "distance_functions"
        distance_functions = {
            "euclidean": self.euclidean_dist,
            "manhattan": self.manhattan_dist,
            "minkowski": self.minkowski_dist,
            "squared_euclidean": self.squared_euclidean_dist,
            "chebyshev": self.chebyshev_dist
        }

        # Sprawdzenie czy podana metryka istnieje
        if self.metric not in distance_functions:
            raise ValueError(f"Unsupported metric: {self.metric}")

        print("threshold: {:.3f}".format(self.threshold))
        predictions = []
        for x_test in X_test:
            # Obliczanie odległości do każdego punktu w zbiorze treningowym
            distances = [distance_functions[self.metric](x_test, x_train) for x_train in self.X_train]
            # Wyznaczenie k najbliższych sąsiadów
            k_indices = np.argsort(distances)[:self.k]
            # Pobranie etykiet k najbliższych sąsiadów
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # Głosowanie większościowe do której klasy przydzielić nowy punkt
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            min_distance = np.min(distances)

            # Sprawdzenie progu odległości
            if min_distance > self.threshold or k_nearest_labels.count(most_common) / self.k < 0.95:
                most_common = max(set(self.y_train)) + 1
            # Dodanie klasy do listy predykcji
            predictions.append(most_common)

        return np.array(predictions)
