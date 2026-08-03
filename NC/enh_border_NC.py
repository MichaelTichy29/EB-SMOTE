import numpy as np
import pandas as pd
from pointdef_NC import generate_sample_cm_NC, generate_sample_em_NC
from distance_NC import compute_median_std, knn_minority, knn_global
from distance_NC import compute_median_std_num
from sklearn.preprocessing import StandardScaler

def generate_dataset_NC(X,
    y,continuous_cols,
    categorical_cols,
    minority_label=1,
    k1=3,
    k2=10,
    samp_meth = "cm",
    #random_state=42
    ):

    rng = np.random.default_rng()#random_state)
    
    
    # ---------------------------------
    # Daten vorbereiten
    # ---------------------------------
    #X_num = X[continuous_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[continuous_cols])
    
    X_cat = X[categorical_cols].astype(str).to_numpy()
    
    y_values = np.asarray(y)
    
            
    #M = compute_median_std(X_num, continuous_cols)
    M = compute_median_std_num(X_num)
    
    minority_idx = np.where(y_values == minority_label)[0]

    synthetic_samples = []
    
    minority_idx = np.where(y_values == minority_label)[0]

    for global_idx in minority_idx:
        
        # -------------------------
        # Minority Nachbarn
        # -------------------------
        neigh_min = knn_minority(global_idx,k1,minority_idx, X_num, X_cat, M)
            
        if len(neigh_min) == 0:
                continue

        pj_idx = rng.choice(neigh_min)
            
        
        # -------------------------
        # mp bestimmen
        # -------------------------
        neigh_p = knn_global(global_idx,k2,X_num,X_cat,M)

        m_p = np.sum(y_values[neigh_p] == minority_label)

        # -------------------------
        # mj bestimmen
        # -------------------------
        neigh_j = knn_global(pj_idx,k2,X_num,X_cat,M)

        m_j = np.sum(y_values[neigh_j] == minority_label)
        
            
        # -------------------------
        # numerische Punkte
        # -------------------------
        p_num = X_num[global_idx]
        pj_num = X_num[pj_idx]
        p_cat = X_cat[global_idx]
        pj_cat = X_cat[pj_idx]
        
        
        # cm oder em 
        if samp_meth == "em":
            pz = generate_sample_em_NC(global_idx, pj_idx, p_num, pj_num, neigh_p, neigh_j, p_cat, pj_cat, m_p, m_j, X_num, X_cat, M, rng)
        else:
            pz = generate_sample_cm_NC(global_idx, pj_idx, p_num, pj_num, neigh_p, neigh_j, p_cat, pj_cat, m_p, m_j, X_num, X_cat, M, rng)
            
            
        if pz is None:
            continue


        synthetic_samples.append(pz)
   
            
           
    # ---------------------------------
    # Rückgabe
    # ---------------------------------
    columns = continuous_cols + categorical_cols

    X_syn = pd.DataFrame(
        synthetic_samples,
        columns=columns
    )

    y_syn = pd.Series(
        np.full(len(X_syn), minority_label)
    )

    X_res = pd.concat(
        [X.reset_index(drop=True), X_syn],
        ignore_index=True
    )

    y_res = pd.concat(
        [pd.Series(y).reset_index(drop=True), y_syn],
        ignore_index=True
    )

    return X_res, y_res    
            
    