from Classification import *
from Read_data import *
from Operations import *

# Utworzenie danych syntetycznych
X,y = synthetic_data()
X = np.array(X)
y = np.array(y)

# Wyznaczenie klas znanych
X_known = X[y<=1]
y_known = y[y<=1]

# Wyznaczenie klas nieznanych
X_unknown = X[y==2]
y_unknown = y[y==2]
#print(y_known)

# Wyznaczenie zbiorów treningowych i testowych
X_train, X_test, y_train, y_test = train_test_sets(X_known,y_known,X_unknown,y_unknown)

