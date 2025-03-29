import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def read_samples():
    n_samp = 700#int(input('enter the number of samples: '))
    n_class = 2#int(input('enter the number of classes: '))
    rand = 40
    X,y = make_classification(n_samples=n_samp,n_classes=n_class,random_state=rand)


    return X,y

def read_banknotes():
    # Wczytanie danych z pliku csv
    df_banknotes = pd.read_csv("data_banknote_auth.csv", header=None)

    # Nadanie nazw kolumnom
    df_banknotes.columns = ["Variance", "Skewness", "Kurtosis", "Entropy", "Class"]

    # Podgląd pierwszych wierszy
    print(df_banknotes)
    return df_banknotes

def read_wines():
    # Lista nazw kolumn (pierwsza kolumna to klasa wina)
    column_names = [
        "Class", "Alcohol", "Malic_Acid", "Ash", "Alcalinity", "Magnesium",
        "Total_Phenols", "Flavanoids", "Nonflav_Phenols", "Proanthocyanins",
        "Color_Intensity", "Hue", "OD280_OD315", "Proline"
    ]

    # Wczytanie danych z pliku csv
    df_wine = pd.read_csv("data_wine.csv", header=None, names=column_names)

    # Podgląd pierwszych wierszy
    print(df_wine)
    return df_wine
