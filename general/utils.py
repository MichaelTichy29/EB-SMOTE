
####################################
#####     Measures      #######
####################################

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score


def evaluate(model, X_test, y_test, beta=2.0):
    # Vorhersageklassen
    y_pred = model.predict(X_test)

    # Wahrscheinlichkeiten für positive Klasse
    y_prob = model.predict_proba(X_test)[:, 1]

    # ROC Curve Werte
    #fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "f_beta": fbeta_score(y_test, y_pred, beta=beta),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),

        # optional: ROC-Daten mit zurückgeben
        #"fpr": fpr.tolist(),
        #"tpr": tpr.tolist(),
        #"thresholds": thresholds.tolist()
    }




def save_results(results):
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    df.to_excel("results.xlsx", index=False)
    
    
def save_results_NC(results_NC):
    df = pd.DataFrame(results_NC)
    df.to_csv("results_NC.csv", index=False)
    df.to_excel("results_NC.xlsx", index=False)



####################################
#####     Grid Search     #######
####################################



import itertools

def expand_grid(grid):
    keys = grid.keys()
    values = grid.values()

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


####################################
#####    Number of synthetic     #######
####################################



def compute_n_synthetic(y, minority_label=1, balance_level=1.0):

    y = np.asarray(y)

    n_min = np.sum(y == minority_label)
    n_maj = np.sum(y != minority_label)

    n_target = int(n_maj * balance_level)
    n_synthetic = max(0, n_target - n_min)

    return n_synthetic



####################################
#####    write results     #######
####################################

import os



def append_result_zw(result, filename):
    df_new = pd.DataFrame([result])

    if not os.path.exists(filename):
        df_new.to_csv(filename, index=False)
        return

    try:
        old_df = pd.read_csv(filename)
    except pd.errors.ParserError:
        raise ValueError(
            f"Die Datei {filename} ist bereits beschädigt. "
            "Bitte einmal löschen und den Lauf neu starten."
        )

    # Alte und neue Spalten zusammenführen
    all_columns = list(old_df.columns)

    for col in df_new.columns:
        if col not in all_columns:
            all_columns.append(col)

    # Fehlende Spalten ergänzen
    for col in all_columns:
        if col not in old_df.columns:
            old_df[col] = None
        if col not in df_new.columns:
            df_new[col] = None

    # Reihenfolge vereinheitlichen
    old_df = old_df[all_columns]
    df_new = df_new[all_columns]

    # Komplett neu schreiben
    result_df = pd.concat([old_df, df_new], ignore_index=True)
    result_df.to_csv(filename, index=False)
