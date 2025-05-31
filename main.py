import numpy as np
from Classification import Knn_with_OSR
from Syntetic_data import X_known, y_known, X_unknown, y_unknown, flag
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from Operations import *
from datetime import datetime


# Lista metryk
metrics = ["euclidean", "manhattan", "minkowski", "squared_euclidean", "chebyshev"]
results_summary = []

print("Wyniki predykcji i metryk dla różnych odległości:\n")
with open("results/results.txt", "w", encoding="utf-8") as f:
    f.write("Wyniki predykcji i metryk dla różnych odległości:\n\n")
    f.write("=" * 60 + "\n")

for metric in metrics:
    scores = []
    for i in range(10):
        print("*************************** Iteracja: {} ***************************".format(i))
        X_train, X_test, y_train, y_test = train_test_sets(X_known, y_known, X_unknown, y_unknown)

        unique_classes = np.unique(np.concatenate((y_train, y_test)))

        class_labels = list(unique_classes)
        label_names = [str(cls) for cls in class_labels]
        label_names[-1] = f"obca {class_labels[-1]}"

        # Inicjalizacja klasyfikatora z daną metryką
        knn = Knn_with_OSR(k=3, threshold=None, metric=metric)

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

        #  Zapis do pliku w tej samej postaci co print
        """
        with open("results/results.txt", "a", encoding="utf-8") as f:
            f.write(f"Threshold: {knn.threshold:.3f}\n")
            f.write(f"Metryka: {metric}\n")
            f.write("Macierz pomyłek:\n")
            f.write("Rzeczywiste / Przewidziane\n")
            f.write(cm_df.to_string())
            f.write(f"\nProcent poprawnych predykcji: {balanced_accuracy:.3f}\n\n")
            f.write("="*60 + "\n")
        """

        #  Zapis do pliku w postaci kodu w LaTeX
        with open("results/results.txt", "a", encoding="utf-8") as f:
            f.write(f"Threshold: {knn.threshold:.3f}\n")
            f.write(f"Metryka: {metric}\n")
            f.write("Macierz pomyłek w formie kodu w LaTeX:\n\n")

            n_rows = len(cm_df)
            n_cols = len(cm_df.columns)

            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Macierz pomyłek metryki {metric}}}\n")
            f.write("\\vspace{10pt}\n")
            f.write(f"\\label{{tab:Macierz pomyłek {metric}}}\n")
            f.write("\\renewcommand{\\arraystretch}{1.3}\n")

            col_format = "|>{\\centering\\arraybackslash}m{0.9cm}"
            col_format += "|" + "|".join([">{\\centering\\arraybackslash}m{2.7cm}"] * n_cols) + "|"
            f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
            f.write("\\hline\n")

            f.write("\\multicolumn{1}{|c|}{} & \\multicolumn{" + str(n_cols) + "}{c|}{\\textbf{Przewidziane}} \\\\\n")
            f.write("\\cline{2-" + str(n_cols + 1) + "}\n")

            header_row = " & " + " & ".join(f"\\textbf{{{col}}}" for col in cm_df.columns) + " \\\\\n"
            f.write("\\multirow{" + str(n_rows) + "}{*}{\\rotatebox{90}{\\textbf{Rzeczywiste}}} " + header_row)
            f.write("\\cline{2-" + str(n_cols + 1) + "}\n")

            for i, (idx, row) in enumerate(cm_df.iterrows()):
                f.write("& ")
                row_data = " & ".join(str(val) for val in row)
                f.write(row_data + " \\\\\n")
                f.write("\\cline{2-" + str(n_cols + 1) + "}\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")

            f.write(f"\nProcent poprawnych predykcji: {balanced_accuracy:.3f}\n\n")
            f.write("=" * 60 + "\n")

        # Utworzenie wykresu dla każdej metryki tylko w iteracji 0
        if flag == 0 and i == 0:
            charts_true_predicted(X_test,y_test,predictions,balanced_accuracy,metric)

    # Wyznaczenie średniej i odchylenia standardowego dla 10 powtórzeń danej metryki
    mean_score = round(np.mean(scores),3)
    std_score = round(np.std(scores),3)
    results_summary.append([metric, round(mean_score, 4), round(std_score, 4)])
    print("\n\033[34mŚrednia balanced accuracy: {:.3f} \033[39m".format(mean_score))
    print("\033[31mOdchylenie standardowe: {:.3f} \033[39m".format(std_score))

df_results = pd.DataFrame(results_summary, columns=["Metryka", "Średnia", "Odchylenie"])
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"results/wyniki_knn_{timestamp}.csv"
df_results.to_csv(filename, index=False)
print(df_results)