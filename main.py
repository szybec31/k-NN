import numpy as np
from Classification import Knn_with_OSR
from Split_class import X_known, y_known, X_unknown, y_unknown, flag
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from Operations import *
from datetime import datetime
import time

start = time.time()

# Lista metryk
metrics = ["euclidean", "manhattan", "minkowski", "squared_euclidean", "chebyshev"]
#metrics = ["euclidean"]
results_summary = []

print("Wyniki predykcji i metryk dla różnych odległości:\n")
with open("results/results.txt", "w", encoding="utf-8") as f:
    f.write("Wyniki predykcji i metryk dla różnych odległości:\n\n")
    f.write("=" * 60 + "\n")

print("X_known shape {}, X_unknown shape {} ".format(X_known.shape,X_unknown.shape))
print("y_known shape {}, y_unknown shape {} ".format(y_known.shape, y_unknown.shape))

for metric in metrics:
    tresholds = []
    scores = []
    for j in range(10):
        random_state=(j+1)*1000
        print("*************************** Iteracja: {}, random_state= {} ***************************".format(j,random_state))
        X_train, X_test, y_train, y_test = train_test_sets(X_known, y_known, X_unknown, y_unknown,random_state)

        unique_classes = np.unique(np.concatenate((y_train, y_test)))

        class_labels = list(unique_classes)
        label_names = [str(cls) for cls in class_labels]
        label_names[-1] = f"obca {class_labels[-1]}"

        # Inicjalizacja klasyfikatora z daną metryką
        knn = Knn_with_OSR(k=5, threshold=None,vote_treshold=None,threshold_percentile=95, metric=metric)

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
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        scores.append(balanced_accuracy)
        print(f"Dokładność poprawnych predykcji: {balanced_accuracy:.3f}\n")

        # Zapis wyników do pliku ---> LateX, dla każdej metryki tylko w iteracji 0
        if j == 0:
            zapis_wyników(metric, knn, cm_df, balanced_accuracy)
        # Utworzenie wykresu dla każdej metryki tylko w iteracji 0
            if flag == 0:
                charts_true_predicted(X_test,y_test,predictions,balanced_accuracy,metric)
    tresholds.append(knn.threshold)
    # Wyznaczenie średniej i odchylenia standardowego dla 10 powtórzeń danej metryki
    mean_threshold = round(np.mean(tresholds),3)
    std_threshold = round(np.std(tresholds),3)
    mean_score = round(np.mean(scores),3)
    std_score = round(np.std(scores),3)

    results_summary.append([metric,"{} ({})".format(mean_threshold,std_threshold),"{} ({})".format(mean_score, std_score)])
    print("\n\033[34mŚrednia balanced accuracy: {:.3f} \033[39m".format(mean_score))
    print("\033[31mOdchylenie standardowe: {:.3f} \033[39m".format(std_score))

f.close()
df_results = pd.DataFrame(results_summary, columns=["Metryka","Treshold", "Wynik"])
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"results/wyniki_knn_{timestamp}.csv"
df_results.to_csv(filename, index=False)
print(df_results)

end = time.time()
print(f"Czas wykonania: {end - start:.5f} sekund")