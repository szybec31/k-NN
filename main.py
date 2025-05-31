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
with open("results/results.txt", "w", encoding="utf-8") as f:
    f.write("Wyniki predykcji i metryk dla różnych odległości:\n\n")
    f.write("=" * 60 + "\n")

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
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    print(f"Procent poprawnych predykcji: {balanced_accuracy:.3f}\n")

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


    # Utworzenie wykresu dla wybranej metryki
    ''' 
   if metric == "euclidean" and flag == 0:
        y_pred = predictions
        X_test1 = X_test
        charts_true_predicted(X_test1,y_test,y_pred,balanced_accuracy,metric)'''

    # Utworzenie wykresu dla każdej metryki
    if flag == 0:
        charts_true_predicted(X_test, y_test, predictions, balanced_accuracy, metric)

