from Read_data import *
from Operations import *

# Wczytanie zbiorów, flaga pozwala rozróżnić wczytany zbiór
X,y,flag = synthetic_data() # flag = 0
#X,y,flag = dry_beans()     # flag = 1
#X,y,flag = sensors()       # flag = 2


X = np.array(X)
y = np.array(y)

# Wyznaczenie klas znanych
X_known = X[y<=1]
y_known = y[y<=1]

# Wyznaczenie klas nieznanych
X_unknown = X[y==2]
y_unknown = y[y==2]
#print(y_known)
