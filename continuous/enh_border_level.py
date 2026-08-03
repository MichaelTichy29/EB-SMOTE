import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pointdef import generate_sample_cm, generate_sample_em

from imblearn.over_sampling.base import BaseOverSampler


def compute_n_synthetic(y, minority_label=1, balance_level=1.0):

    y = np.asarray(y)

    n_min = np.sum(y == minority_label)
    n_maj = np.sum(y != minority_label)

    n_target = int(n_maj * balance_level)
    n_synthetic = max(0, n_target - n_min)

    return n_synthetic


def generate_dataset_level(X, y, minority_label=1, k1=3, k2=10, samp_meth="cm", balance_level=1.0): #random_state=42, 

    rng = np.random.default_rng() #random_state)

    X_values = np.asarray(X)
    y_values = np.asarray(y)

    minority_idx = np.where(y_values == minority_label)[0]
    
    
    X_min = X_values[minority_idx]


    # Minority neighbors
    nn_min = NearestNeighbors(n_neighbors=k1 + 1).fit(X_min)
    _, ind_min = nn_min.kneighbors(X_min)
    ind_min = ind_min[:, 1:]

    # Global neighbors
    nn_all = NearestNeighbors(n_neighbors=k2 + 1).fit(X_values)

    n_synthetic_target = compute_n_synthetic(y_values, minority_label, balance_level)
    synthetic_samples = []
    

    while len(synthetic_samples) < n_synthetic_target:
        
        generated_in_round = False

        
        for row_pos, global_idx in enumerate(minority_idx):
            p = X_values[global_idx]
    
               
            chosen_local = rng.choice(ind_min[row_pos])
            pj_idx = minority_idx[chosen_local]
            pj = X_values[pj_idx]
    
            # neighbors von p
            _, neigh_p = nn_all.kneighbors(p.reshape(1, -1))
            neigh_p = neigh_p[0][1:]
            mp = np.sum(y_values[neigh_p] == minority_label)
    
            # neighbors von pj
            _, neigh_j = nn_all.kneighbors(pj.reshape(1, -1))
            neigh_j = neigh_j[0][1:]
            mj = np.sum(y_values[neigh_j] == minority_label)
    
            # mein Regelwerk
            if samp_meth == "em":
                pz = generate_sample_em(p, pj, mp, mj, nn_all, X_values, rng)
            elif samp_meth == "cm":
                pz = generate_sample_cm(p, pj, mp, mj, nn_all, X_values, rng)
                               
            if pz is not None:
                synthetic_samples.append(pz)
                generated_in_round = True
            
            if len(synthetic_samples) >= n_synthetic_target:
                print("Anzahl syn inst. = ", len(synthetic_samples))
                break
            
            
        if not generated_in_round:
            print("Keine neuen Samples mehr möglich.")
            print("Anzahl syn inst. = ", len(synthetic_samples))
            break       
        
    print("Anzahl syn inst. = ", len(synthetic_samples))
    if len(synthetic_samples) == 0:
        return X_values, y_values
    else:
        X_syn = np.array(synthetic_samples)
        y_syn = np.full(len(X_syn), minority_label)
    
        X_res = np.vstack([X_values, X_syn])
        y_res = np.hstack([y_values, y_syn])

    return X_res, y_res

##########################################

class EnhancedBorderline(BaseOverSampler):

    _sampling_type = "over-sampling"

    def __init__(
        self,
        sampling_strategy="auto",
        minority_label=1,
        k1=3,
        k2=20,
        samp_meth="cm",
        #random_state=42,
        balance_level=1.0
    ):

        self.sampling_strategy = sampling_strategy
        self.minority_label = minority_label
        self.k1 = k1
        self.k2 = k2
        self.samp_meth = samp_meth
        #self.random_state = random_state
        self.balance_level = balance_level


    def _fit_resample(self, X, y):

        X_res, y_res = generate_dataset_level(
            X=X,
            y=y,
            minority_label=self.minority_label,
            k1=self.k1,
            k2=self.k2,
            samp_meth=self.samp_meth,
            #random_state=self.random_state,
            balance_level=self.balance_level
        )

        return np.asarray(X_res), np.asarray(y_res)