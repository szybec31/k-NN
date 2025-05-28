import numpy as np
from Classification import Knn_with_OSR
from Syntetic_data import X_train, X_test, y_train, y_test, flag
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from Operations import *

# Lista metryk
metrics = ["euclidean", "manhattan", "minkowski", "squared_euclidean", "chebyshev"]

unique_classes = np.unique(np.concatenate((y_train, y_test)))

class_labels = list(unique_classes)
label_names = [str(cls) for cls in class_labels]
label_names[-1] = f"obca {class_labels[-1]}"


print("Wyniki predykcji i metryk dla różnych odległości:\n")

for metric in metrics:
    print("=" * 60)
    # Inicjalizacja klasyfikatora z daną metryką
    knn = Knn_with_OSR(k=3, metric=metric)

    # Trenowanie modelu
    knn.fit(X_train, y_train)

    # Predykcja
    predictions = knn.predict(X_test)

    # Macierz pomyłek jako DataFrame
    cm = confusion_matrix(y_test, predictions, labels=class_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Klasa {name}" for name in label_names],
        columns=[f"Klasa {name}" for name in label_names]
    )
    print("Metryka: {}".format(metric))
    print("Macierz pomyłek:")
    print("rzeczywiste/przewidziane")
    print(cm_df)

    # Dokładność
    balanced_accuracy = balanced_accuracy_score(y_test,predictions)
    print(f"Procent poprawnych predykcji: {balanced_accuracy:.3f}\n")

    # Utworzenie wykresu dla wybranej metryki
    ''' 
   if metric == "euclidean" and flag == 0:
        y_pred = predictions
        X_test1 = X_test
        charts_true_predicted(X_test1,y_test,y_pred,balanced_accuracy,metric)'''

    # Utworzenie wykresu dla każdej metryki
    if flag == 0:
        charts_true_predicted(X_test,y_test,predictions,balanced_accuracy,metric)