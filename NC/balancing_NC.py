
####################################
#####     Oversampling     #######
####################################

"""
from imblearn.over_sampling import SMOTE, RandomOverSampler

def apply_balancing(X_train, y_train, method):
    if method == "smote":
        sampler = SMOTE()
    elif method == "random_oversample":
        sampler = RandomOverSampler()
    else:
        return X_train, y_train

    return sampler.fit_resample(X_train, y_train)
"""

from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import RandomOverSampler

import pandas as pd
import numpy as np
#from enh_border_level_NC import generate_dataset_level_NC
from enh_border_level_NC import EnhancedBorderlineNC


def apply_balancing_NC(X, y, method_name, params, fallback=None):
   
    balance_level = params.get("balance_level", 1.0)
    # =========================================
    # 0) INPUT NORMALISIEREN
    # =========================================
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Index synchronisieren
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # =========================================
    # 1) RANDOM OVERSAMPLING
    # =========================================
    if method_name == "random_oversample":

        sampler = RandomOverSampler(sampling_strategy=balance_level)#, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)

    # =========================================
    # 2) SMOTENC
    # =========================================
    elif method_name == "smote_nc":

        categorical_cols = params["categorical_cols"]

        # kategorien sind int, aber intern weiter logische kat.
        for col in categorical_cols:
            X[col] = X[col].astype("category")
       
        # -----------------------------
        # 2.5 categorical index bestimmen
        # -----------------------------
        cat_idx = [X.columns.get_loc(c) for c in categorical_cols]

        # -----------------------------
        # 2.6 SMOTENC
        # -----------------------------
        sampler = SMOTENC(
            categorical_features=cat_idx,
            sampling_strategy=balance_level,
            k_neighbors=params.get("k_neighbors", 5),
            #random_state=42
        )

        X_res, y_res = sampler.fit_resample(X, y)

        # -----------------------------
        # 2.7 zurück zu DataFrame
        # -----------------------------
        X_res = pd.DataFrame(X_res, columns=X.columns)
        y_res = pd.Series(y_res)

        return X_res, y_res
    
    # =========================================
    # 3) DEIN CUSTOM SAMPLER
    # =========================================
    
    elif method_name == "enhance_borderline_NC":

        sampler = EnhancedBorderlineNC(
            continuous_cols=params["continuous_cols"],
            categorical_cols=params["categorical_cols"],
            k1=params.get("k1", 3),
            k2=params.get("k2", 10),
            balance_level=params.get("balance_level", 1.0),
            #random_state=42
        )

        X_res, y_res = sampler.fit_resample(X, y)

        # -----------------------------
        # 2.7 zurück zu DataFrame
        # -----------------------------
        X_res = pd.DataFrame(X_res, columns=X.columns)
        y_res = pd.Series(y_res)

        return X_res, y_res
    
    # =========================================
    # 4) FALLBACK
    # =========================================
    else:
        if fallback == "random":
            sampler = RandomOverSampler()#random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)

        raise ValueError(f"Unbekannte Methode: {method_name}")