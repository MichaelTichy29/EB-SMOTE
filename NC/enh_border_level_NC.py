import numpy as np
import pandas as pd
from pointdef_NC import generate_sample_cm_NC, generate_sample_em_NC
from distance_NC import compute_median_std, knn_minority, knn_global
from distance_NC import compute_median_std_num
from sklearn.preprocessing import StandardScaler

def compute_n_synthetic(y, minority_label=1, balance_level=1.0):

    y = np.asarray(y)

    n_min = np.sum(y == minority_label)
    n_maj = np.sum(y != minority_label)

    n_target = int(n_maj * balance_level)
    n_synthetic = max(0, n_target - n_min)

    return n_synthetic


def generate_dataset_level_NC(X, y, continuous_cols, categorical_cols, minority_label=1, k1=3, k2=10, samp_meth="cm", balance_level=1.0): #random_state=42,

    # -----------------------------
    # INPUT NORMALISIEREN
    # -----------------------------
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=continuous_cols + categorical_cols)

    X = X.copy()
    y = pd.Series(y).reset_index(drop=True)

    # -----------------------------
    # TYPES FIXEN (WICHTIG!)
    # -----------------------------
    X[continuous_cols] = X[continuous_cols].apply(pd.to_numeric, errors="coerce")
    X[categorical_cols] = X[categorical_cols].astype(str)

    

    rng = np.random.default_rng()#random_state)

    # ---------------------------------
    # Daten vorbereiten
    # ---------------------------------
    
    
    X_num = X[continuous_cols].to_numpy()
    X_cat = X[categorical_cols].astype(str).to_numpy()
    
    y_values = np.asarray(y)
    
    
    M = compute_median_std_num(X_num)
    
    minority_idx = np.where(y_values == minority_label)[0]

    
    global_cache = {}
    m_cache = {}
    for i in range(len(X)):
        neigh = knn_global(i, k2, X_num, X_cat, M)
        global_cache[i] = neigh
        m_cache[i] = np.sum(y_values[neigh] == minority_label)
        
    minor_cache = {}
    for i in minority_idx:
        minor_cache[i] = knn_minority(i, k1, minority_idx, X_num, X_cat,M)



    n_synthetic_target = compute_n_synthetic(y_values, minority_label, balance_level)
    #synthetic_samples = []
    synthetic_num = []
    synthetic_cat = []
    
    
    while len(synthetic_num) < n_synthetic_target:
        
        generated_in_round = False

        
        for global_idx in minority_idx:
                    
            neighbors_min = minor_cache[global_idx]

            pj_idx = rng.choice(neighbors_min)
    
            neigh_p = global_cache[global_idx]
            m_p = m_cache[global_idx]

            neigh_j = global_cache[pj_idx]
            m_j = m_cache[pj_idx] 
            # -------------------------
            # numerische Punkte
            # -------------------------
            p_num = X_num[global_idx]
            pj_num = X_num[pj_idx]
            p_cat = X_cat[global_idx]
            pj_cat = X_cat[pj_idx]
           
           
            # cm oder em 
            if samp_meth == "em":
                result = generate_sample_em_NC(global_idx, pj_idx, p_num, pj_num, neigh_p, neigh_j, p_cat, pj_cat, m_p, m_j, X_num, X_cat, M, rng)
            else:
                result = generate_sample_cm_NC(global_idx, pj_idx, p_num, pj_num, neigh_p, neigh_j, p_cat, pj_cat, m_p, m_j, X_num, X_cat, M, rng)
               
               
            if result is None:
                continue
            
            pz_num, pz_cat = result
            synthetic_num.append(pz_num)
            synthetic_cat.append(pz_cat)
            
            generated_in_round = True            
            
            if len(synthetic_num) >= n_synthetic_target:
                break
            
        
        if not generated_in_round:
            print("Keine neuen Samples mehr möglich.")
            print("Anzahl syn inst. = ", len(synthetic_num))
            break     
    
    print("Anzahl syn inst. = ", len(synthetic_num))
    X_orig = X.copy().reset_index(drop=True)
    y_orig = pd.Series(y).reset_index(drop=True)
    
        
    if len(synthetic_num) == 0:
        X_res = X_orig
        y_res = y_orig
    else:
        X_num_syn = np.asarray(synthetic_num, dtype=float)
        X_cat_syn = np.asarray(synthetic_cat, dtype=object)
    
        X_num_df = pd.DataFrame(X_num_syn, columns=continuous_cols)
        X_cat_df = pd.DataFrame(X_cat_syn, columns=categorical_cols).astype(str)
    
        X_syn = pd.concat([X_num_df, X_cat_df], axis=1)
    
        X_res = pd.concat([X_orig, X_syn], ignore_index=True)
        y_res = pd.concat([y_orig, pd.Series([minority_label] * len(X_syn))], ignore_index=True)

    
    return X_res, y_res



from imblearn.over_sampling.base import BaseOverSampler


class EnhancedBorderlineNC(BaseOverSampler):

    _sampling_type = "over-sampling"

    def __init__(
        self,
        continuous_cols,
        categorical_cols,
        sampling_strategy="auto",
        minority_label=1,
        k1=3,
        k2=20,
        samp_meth="cm",
        #random_state=42,
        balance_level=1.0
    ):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.sampling_strategy = sampling_strategy
        self.minority_label = minority_label
        self.k1 = k1
        self.k2 = k2
        self.samp_meth = samp_meth
        #self.random_state = random_state
        self.balance_level = balance_level


    def fit(self, X, y):
        """
        Hier merken wir uns die Spaltenreihenfolge!
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            raise ValueError("X muss ein DataFrame sein beim fit()")

        return self
    
    def _fit_resample(self, X, y):

       # ---------------------------------
       # 1. DataFrame wiederherstellen
       # ---------------------------------
       if not hasattr(self, "feature_names_in_"):
           raise ValueError("fit() wurde nicht korrekt aufgerufen")

       X = pd.DataFrame(X, columns=self.feature_names_in_)
       y = pd.Series(y).reset_index(drop=True)

       # ---------------------------------
       # 2. Oversampling
       # ---------------------------------
       X_res, y_res = generate_dataset_level_NC(
           X=X,
           y=y,
           continuous_cols=self.continuous_cols,
           categorical_cols=self.categorical_cols,
           minority_label=self.minority_label,
           k1=self.k1,
           k2=self.k2,
           samp_meth=self.samp_meth,
           #random_state=self.random_state,
           balance_level=self.balance_level
       )

       # ---------------------------------
       # 3. Zurück zu numpy (Pipeline erwartet das)
       # ---------------------------------
       return X_res.to_numpy(), y_res.to_numpy()