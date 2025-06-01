from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

    # Rozłożenie klasy znanych i nieznanych po równo w zbiorze testowym - wybiera losowe próbki z unknown
    num_known = X_test_known.shape[0]
    if X_unknown.shape[0] > num_known:
        idx = np.random.choice(X_unknown.shape[0], num_known, replace=False)
        X_unknown = X_unknown[idx]
        y_unknown = y_unknown[idx]

    # Ostateczna postać zbioru testowego
    X_test_final = np.vstack([X_test_known, X_unknown])
    y_test_final = np.hstack([y_test_known, y_unknown])

    # sprawdzenie kształtu tablic
    print("X_train_known shape {}".format(X_train_known.shape))
    print("X_test_known shape {}, X_unknown shape {} ".format(X_test_known.shape,X_unknown.shape))
    #print("y_test_known shape {}, y_unknown shape {} ".format(y_test_known.shape, y_unknown.shape))
    #print("Nowa etykieta klasy nieznanej:", unknown_label)
    # Zwraca potrzebne zbiory danych
    return X_train_known,X_test_final,y_train_known,y_test_final

# Utworzenie wykresu porównania etykiet rzeczywistych i przewidzianych - dla zbioru syntetycznego
def charts_true_predicted(X_test,y_test,y_pred,balanced_accuracy,metric):
    colors = ["gray","green","blue"]
    labels = ['Klasa 0', 'Klasa 1', 'Nieznana klasa']

    # Mapowanie klasy do koloru
    class_to_color = {i: colors[i] for i in range(3)}

    # Kolory przypisane do punktów
    color_true = [class_to_color[label] for label in y_test]
    color_pred = [class_to_color[label] for label in y_pred]

    fig, ax = plt.subplots(2, 1,figsize=(10,7))
    plt.suptitle("Metryka: "+metric)
    ax[0].scatter(X_test[:, 0], X_test[:, 1], c=color_true)
    ax[0].set_title("Prawdziwe etykiety")
    ax[0].set_xlabel("cecha x")
    ax[0].set_ylabel("cecha y")

    ax[1].set_title("Predykcja etykiet: {:.3f}".format(balanced_accuracy))
    ax[1].set_xlabel("cecha x")
    ax[1].set_ylabel("cecha y")
    ax[1].scatter(X_test[:, 0], X_test[:, 1], c=color_pred)

    # Legenda
    legend_patches = [Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1, 0.5), title="Klasy")

    # Zapisanie wykresów do png
    plt.tight_layout(rect=[0, 0, 0.83, 0.95])
    plt.savefig("results/{}_True_Predict.png".format(metric))
    plt.close(fig)

    #plt.show()

def zapis_wyników(metric,knn,cm_df,balanced_accuracy):
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