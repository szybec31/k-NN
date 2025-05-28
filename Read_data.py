import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def synthetic_data():
    # Wygenerowanie Danych syntetycznych
    X, y = make_classification(
                                n_samples=1200,
                                n_features=2,
                                n_informative=2,
                                n_redundant=0,
                                n_classes=3,
                                n_clusters_per_class=1,
                                class_sep=1.5,
                                random_state=1)
    X = X.astype(float)
    y = y.astype(int)
    # Wyświetlenie danych syntetycznych na wykresie
    X0 = X[y == 0]
    X1 = X[y == 1]
    X2 = X[y == 2]

    plt.title("Dane Syntetyczne")
    plt.xlabel("oś pozioma x")
    plt.ylabel("oś pionowa y")
    plt.scatter(X0[:, 0], X0[:, 1], c="gray", label="Klasa 0")
    plt.scatter(X1[:, 0], X1[:, 1], c="green", label="Klasa 1")
    plt.scatter(X2[:, 0], X2[:, 1], c="blue", label="Klasa 2")
    plt.legend()
    plt.savefig("Syntetyczne.png")
    plt.show()

    return X, y

def dry_beans():
    data = pd.read_csv("datasets/Dry_Bean.csv", delimiter=";")
    # print(data.values[:, -1])
    # print(data.shape)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
# dry_beans()

def steel_plates_defection():
    data = pd.read_csv("datasets/Steel.csv", delimiter=";")
    # print(data.values[:,-1])
    # print(data.shape)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
# steel_plates_defection()


def sensors():
    data = pd.read_csv("datasets/Sensorless_drive_diagnosis.csv", delimiter=";")
    # print(data.values[:,-1])
    # print(data.shape)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
# sensors()


