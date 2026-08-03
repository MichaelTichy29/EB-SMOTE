####################################
#####     Pipeline - Kernstueck     #######
####################################



from data1 import preprocess
from models import train_model
from balancing import apply_balancing
from utils import evaluate
from utils import expand_grid
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from utils import append_result_zw




def cross_validate_model(
    X,
    y,
    algorithm,
    params,
    method_name,
    method_params,
    fallback,
    config
):

    skf = StratifiedKFold(
        n_splits=config["cv"]["folds"],
        shuffle=True,
        #random_state=42
    )

    scores = []
    num_syn = 0

    for train_idx, test_idx in skf.split(X, y):

        # ---------------------------------
        # pandas indexing
        # ---------------------------------
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        # ---------------------------------
        # Balancing NUR auf Train
        # ---------------------------------
        
       
        
        X_bal, y_bal = apply_balancing(
            X_train,
            y_train,
            method_name,
            method_params,
            fallback
        )

        num_syn_akt = len(y_bal) - len(y_train)
        # Modell trainieren 
        # ---------------------------------
        model = train_model(X_bal, y_bal, algorithm, params)

        # ---------------------------------
        # Evaluieren
        # ---------------------------------
        beta = config["metrics"].get("f_beta", {}).get("beta", 2.0)
        fold_scores = evaluate(model, X_test, y_test, beta=beta)

        scores.append(fold_scores)
        num_syn = num_syn + num_syn_akt
    # ---------------------------------
    # Mittelwerte
    # ---------------------------------
    avg_scores = {
        metric: np.mean([s[metric] for s in scores])
        for metric in scores[0]
    }

    # ---------------------------------
    # Standardabweichungen
    # ---------------------------------
    for metric in scores[0]:
        avg_scores[f"{metric}_std"] = np.std(
            [s[metric] for s in scores]
        )

    return avg_scores, num_syn


###########################


#################
def run_all(config):

    results = []

    for dataset in config["data"]["datasets"]:

        print(f"\n=== Dataset: {dataset['name']} ===")

        # ---------------------------------
        # Daten laden
        # ---------------------------------
        df = pd.read_csv(dataset["path"])
        
        #print("Shape vor frac  = ", df.shape)
        #print("value counts vor frac  = ", df["Class"].value_counts())
        
        # Optional Sampling
        if "sample_frac" in dataset:
            df = df.sample(
                frac=dataset["sample_frac"],
                random_state=42
            )
        #print("Shape vor prec  = ", df.shape)
        #print("value counts vor prec  = ", df["Class"].value_counts())
            

        # ---------------------------------
        # pandas behalten!
        # ---------------------------------
        X = df.drop(config["data"]["target"], axis=1)
        y = df[config["data"]["target"]]

        # ---------------------------------
        # Preprocessing
        # ---------------------------------
        X, y = preprocess(X, y, config)
        
        #print("Shape input  = ", df.shape)
        #print("value counts input  = ", df["Class"].value_counts())
        

        balancing_methods = config["balancing"]["methods"]

        for method in balancing_methods:

            if not method.get("enabled", True):
                continue

            method_name = method["name"]
            fallback = method.get("fallback")

            # ---------------------------------
            # Parametergrid
            # ---------------------------------
            method_param_list = [{}]

            if "grid" in method:
                method_param_list = list(
                    expand_grid(method["grid"])
                )

            for grid_params in method_param_list:

                # combine static + grid
                method_params = {**grid_params}

                for exp in config["experiments"]:

                    for params in expand_grid(exp["grid"]):

                        print(
                            f"{dataset['name']} | "
                            f"{exp['name']} | "
                            f"{method_name} | "
                            f"{method_params} | "
                            f"{params}"
                        )

                        scores , num_syn = cross_validate_model(
                            X,
                            y,
                            exp["algorithm"],
                            params,
                            method_name,
                            method_params,
                            fallback,
                            config
                        )
                        
                        #scores = {k: round(v, 4) for k, v in scores.items()}
                        scores = {
                            k: f"{v:.4f}" if isinstance(v, (int, float)) else v
                            for k, v in scores.items()
                        }
                        results.append({
                            "dataset": dataset["name"],
                            "model": exp["name"],
                            "balancing": method_name,
                            **method_params,
                            **params,
                            **scores
                        })
                        
                        # fuer zwischenspeicher
                        result_zw = {
                            "dataset": dataset["name"],
                            "model": exp["name"],
                            "balancing": method_name,
                            **method_params,
                            **params,
                            **scores, 
                            "num syn" : num_syn
                        }

                        append_result_zw(result_zw, "results_zw.csv")
                        


    return results


