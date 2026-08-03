
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

import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler,  BorderlineSMOTE
from imblearn.over_sampling import ADASYN
import smote_variants as sv
#from enh_border_level import generate_dataset_level
from enh_border_level import EnhancedBorderline


def apply_balancing(X, y, method_name, params=None, fallback=None):
    params = params or {}

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = None

    
    if method_name == "smote":
        class_counts = Counter(y)
        min_class = min(class_counts.values())

        if min_class < 2:
            return X, y  # SMOTE nicht möglich

        #k_neighbors = min(5, min_class - 1)
        k_neighbors = min(params.get("k_neighbors", 5), min_class - 1)
        sampler = SMOTE(k_neighbors=k_neighbors)


    elif method_name == "random_oversample":
        sampler = RandomOverSampler()
        
    elif method_name == "adasyn":
        class_counts = Counter(y)
        min_class = min(class_counts.values())
    
        if min_class < 2:
            return X, y
    
        n_neighbors = min(params.get("n_neighbors", 5), min_class - 1)
    
        #sampler = ADASYN(n_neighbors=n_neighbors)
        try:
            sampler = ADASYN(n_neighbors=n_neighbors)
            return sampler.fit_resample(X, y)

        except RuntimeError:
            if fallback == "smote":
                sampler = SMOTE()
                return sampler.fit_resample(X, y)
            
    elif method_name == "borderline_smote":
        class_counts = Counter(y)
        min_class = min(class_counts.values())

        if min_class < 2:
            return X, y

        k_neighbors = min(params.get("k_neighbors", 5), min_class - 1)

        sampler = BorderlineSMOTE(
            kind=params.get("kind", "borderline-1"),
            k_neighbors=k_neighbors
        )
        
    elif method_name == "sl_smote":
        class_counts = Counter(y)
        min_class = min(class_counts.values())

        if min_class < 2:
            return X, y

        k_neighbors = min(params.get("k_neighbors", 5), min_class - 1)

        sampler = sv.Safe_Level_SMOTE(
            k_neighbors=k_neighbors
        )
        
    elif method_name == "enhance_borderline":
        sampler = EnhancedBorderline(
            k1=params.get("k1", 3),
            k2=params.get("k2", 10),
            samp_meth=params.get("samp_meth", "cm"),
            balance_level=params.get("balance_level", 1.0),
            #random_state=42
        )
                   
    
    else:
        return X, y

    X_in = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y_in = y.to_numpy() if isinstance(y, pd.Series) else y
    
    X_res, y_res = sampler.fit_resample(X_in, y_in)

    if feature_names is not None:
        X_res = pd.DataFrame(X_res, columns=feature_names)
    else:
        X_res = pd.DataFrame(X_res)

    y_res = pd.Series(y_res)

    return X_res, y_res

    #return sampler.fit_resample(X, y)