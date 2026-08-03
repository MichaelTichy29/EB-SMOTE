####################################
#####     Pipeline - Kernstueck     #######
####################################



from data1 import preprocess_NC
from models import train_model
from balancing_NC import apply_balancing_NC
from utils import evaluate
from utils import expand_grid
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from utils import append_result_zw


def cross_validate_model(X, y, algorithm, params, method_name, method_params, fallback, config):
    skf = StratifiedKFold(
        n_splits=config["cv"]["folds"],
        shuffle=True,
        #random_state=42
    )

    scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_bal, y_bal = apply_balancing_NC(X_train, y_train, method_name, method_params, fallback)
        
        X_train_enc = pd.get_dummies(X_bal)
        X_test_enc = pd.get_dummies(X_test)
        X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

        
        model = train_model(X_train_enc, y_bal, algorithm, params)

        fold_scores = evaluate(model, X_test_enc, y_test)
        scores.append(fold_scores)

    # MIttelwert 
    avg_scores = {
        metric: np.mean([s[metric] for s in scores])
        for metric in scores[0]
    }
    
    #std Abweichung
    for metric in scores[0]:
        avg_scores[f"{metric}_std"] = np.std([s[metric] for s in scores])

    return avg_scores


###########################


#################

def run_all_NC(config):
    results = []

    for dataset in config["data"]["datasets"]:
    

        print(f"\n=== Dataset: {dataset['name']} ===")

        # Daten laden
        df = pd.read_csv(dataset["path"])
       
        #print("Shape vor frac  = ", df.shape)
        #print("value counts vor frac  = ", df["Class"].value_counts())
       
        # Optional Sampling
        if "sample_frac" in dataset:
            df = df.sample(frac=dataset["sample_frac"], random_state=42)
            
        #print("Shape vor prec  = ", df.shape)
        #print("value counts vor prec  = ", df["Class"].value_counts())
    
        X = df.drop(config["data"]["target"], axis=1)
        y = df[config["data"]["target"]]
       
        X, y = preprocess_NC(X,y,config)
        
        #print("Shape input  = ", df.shape)
        #print("value counts input  = ", df["Class"].value_counts())
        
        balancing_methods = config["balancing"]["methods"]

        for method in balancing_methods:  
            if not method.get("enabled", True):
                continue

            method_name = method["name"]
            fallback = method.get("fallback")
        
            # Default
            method_param_list = [{}]
        
            # Grid
            if "grid" in method:
                method_param_list = list(expand_grid(method["grid"]))
            # Test Test Test Test    
            print(method_param_list[:3])    
        
            for grid_params in method_param_list:
        
                # combine config + grid
                method_params = {
                    **grid_params,
                    "continuous_cols": method.get("continuous_cols", []),
                    "categorical_cols": method.get("categorical_cols", [])
                }
        
                for exp in config["experiments"]:
                    if not exp.get("enabled", True):
                        continue
        
                    for params in expand_grid(exp["grid"]):
        
                        print(f"{dataset['name']} | {exp['name']} | {method_name} | {method_params} | {params}")
        
                        scores = cross_validate_model(
                            X, y,
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
                            **scores
                        }

                        append_result_zw(result_zw, "results_nc_zw.csv")    
        return results


