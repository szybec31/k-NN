import numpy as np
from Read_data import *
from Operations import *

# Wczytanie zbiorów, flaga pozwala rozróżnić wczytany zbiór
X,y,flag = synthetic_data()             # flag = 0
#X,y,flag = dry_beans()                 # flag = 1
#X,y,flag = letter_recognition()        # flag = 2
#X,y,flag = penDigits()                 # flag = 3

X = np.array(X)
y = np.array(y)

classes, counts = np.unique(y,return_counts=True)
for cls, count in zip(classes,counts):
    print(f"Klasa {cls}: {count} próbek")



# Wyznaczenie klas znanych
mask = y < 2
#mask = (y == 0) | (y == 3)

X_known = X[mask]
y_known = y[mask]

# Wyznaczenie klas nieznanych
X_unknown = X[y == 2]
y_unknown = y[y == 2]
#print(y_known)
