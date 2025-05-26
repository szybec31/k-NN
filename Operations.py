from sklearn.model_selection import train_test_split
import numpy as np

# Utworzenie danych treningowych - znane klasy, oraz zbiór testowy - X_test_known + X_unknwon
# Jest to modyfikacja train_test_split jako dostosowanie do OSR
def train_test_sets(X_known,y_known,X_unknown,y_unknown):
    X_train_known, X_test_known, y_train_known, y_test_known = train_test_split(
        X_known, y_known,
        test_size=0.2,  # 20% do testu
        stratify=y_known)  # zachowanie proporcji klas
        #random_state=42)  # powtarzalność

    unknown_label = y_known.max() + 1
    y_unknown = np.full_like(y_unknown, fill_value=unknown_label)

    # Ostateczna postać zbioru testowego
    X_test_final = np.vstack([X_test_known, X_unknown])
    y_test_final = np.hstack([y_test_known, y_unknown])

    # sprawdzenie krztałtu tablic
    print("X_test_known shape {}, X_unknown shape {} ".format(X_test_known.shape,X_unknown.shape))
    print("y_test_known shape {}, y_unknown shape {} ".format(y_test_known.shape, y_unknown.shape))
    print("Nowa etykieta klasy nieznanej:", unknown_label)
    # Zwraca potrzebne zbiory danych
    return X_train_known,X_test_final,y_train_known,y_test_final
