from distance_NC import knn_global


# -------------------------------------------------
# Eingaben pro Seedpunkt bereits vorhanden:
# p        -> numpy array des Seedpunkts
# pj       -> numpy array des gewählten Nachbarn
# m_p      -> minority count von p
# m_j      -> minority count von pj
# nn_all   -> trainiertes NearestNeighbors Modell auf gesamtem Datensatz
# X_values -> komplette Feature-Matrix als numpy array
# rng      -> np.random.default_rng(...)
# -------------------------------------------------

# For the categorial values 
from collections import Counter
import numpy as np


def choose_category_from_neighbors(neighbor_idx, X_cat, col):
    values = X_cat[neighbor_idx, col]

    counts = Counter(values)

    max_count = max(counts.values())

    top_vals = {
        v for v, c in counts.items()
        if c == max_count
    }

    # eindeutig
    if len(top_vals) == 1:
        return next(iter(top_vals))

    # tie-break:
    # nächster Nachbar gewinnt
    for idx in neighbor_idx:
        val = X_cat[idx, col]
        if val in top_vals:
            return val


def generate_sample_cm_NC(p_idx, pj_idx, p_num, pj_num, neigh_p, neigh_j, p_cat, pj_cat, m_p, m_j, X_num, X_cat, M, rng):
    
    # -------------------------------------------------
    # Regel 1
    # -------------------------------------------------
    if m_p == 0 and m_j == 0:
        
        return None

    # -------------------------------------------------
    # Regel 2
    # -------------------------------------------------
    elif m_p == 0 and m_j > 0:
        
        neighbors = knn_global(pj_idx, 1, X_num, X_cat, M)

        z_idx = neighbors[0]               # erster echter Nachbar
        x_z = X_num[z_idx]

        r = rng.uniform(0.0, 0.5)

        pz_num = pj_num + r * (x_z - pj_num)
        
    # -------------------------------------------------
    # Regel 3
    # -------------------------------------------------
    elif m_p > 0 and m_j == 0:
        
        neighbors = knn_global(p_idx, 1, X_num, X_cat, M)

        z_idx = neighbors[0]               # erster echter Nachbar
        x_z = X_num[z_idx]

        r = rng.uniform(0.0, 0.5)

        pz_num = p_num + r * (x_z - p_num)
          
    # -------------------------------------------------
    # Regel 4
    # -------------------------------------------------
    elif m_p == m_j:
        
        r = rng.uniform(0.0, 1.0)

        pz_num = p_num + r * (pj_num - p_num)
        
        
    # -------------------------------------------------
    # Regel 5
    # -------------------------------------------------
    elif m_p < m_j:
        
        t = m_j / (m_p + m_j)
        r = rng.uniform(t, 1.0)

        pz_num = p_num + r * (pj_num - p_num)
        
        
    # -------------------------------------------------
    # Regel 6
    # -------------------------------------------------
    else:   # m_j < m_p
        
        t = m_j / (m_p + m_j)
        r = rng.uniform(0.0, t)

        pz_num = p_num + r * (pj_num - p_num)
        
  # -------------------------
    # kategorialer Teil
    # -------------------------
    if m_j <= m_p:
        ref_neighbors = neigh_p
    else:
        ref_neighbors = neigh_j

    pz_cat = np.array([
        choose_category_from_neighbors(ref_neighbors, X_cat, col)
        for col in range(X_cat.shape[1])
    ])

    # -------------------------
    # zusammenführen
    # -------------------------
    #pz = np.concatenate([np.asarray(pz_num).reshape(-1), np.asarray(pz_cat).reshape(-1)]).astype(object)
    
    return pz_num.astype(float), pz_cat.astype(str)

#####################################################################




def generate_sample_em_NC(p_idx, pj_idx, p_num, pj_num, neigh_p, neigh_j, p_cat, pj_cat, m_p, m_j, X_num, X_cat, M, rng):
   
    
    # -------------------------------------------------
    # Regel 1
    # -------------------------------------------------
    if m_p == 0 or m_j == 0:
        return None

    # -------------------------------------------------
    # Regel 2
    # -------------------------------------------------
    

    # -------------------------------------------------
    # Regel 3
    # -------------------------------------------------
    
    # -------------------------------------------------
    # Regel 4 = same
    # -------------------------------------------------
    elif m_p == m_j:

        r = rng.uniform(0.0, 1.0)

        pz_num = p_num + r * (pj_num - p_num)
        
    # -------------------------------------------------
    # Regel 5
    # -------------------------------------------------
    elif m_p < m_j:

        t = m_p / (m_p + m_j)
        r = rng.uniform(0.0, t)

        pz_num = p_num + r * (pj_num - p_num)
        
    # -------------------------------------------------
    # Regel 6
    # -------------------------------------------------
    else:   # m_j < m_p

        t = m_p / (m_p + m_j)
        r = rng.uniform(t, 1.0)

        pz_num = p_num + r * (pj_num - p_num)

        
        
    # -------------------------
    # kategorialer Teil
    # -------------------------
    if m_j <= m_p:
        ref_neighbors = neigh_j
    else:
        ref_neighbors = neigh_p

    pz_cat = np.array([
        choose_category_from_neighbors(ref_neighbors, X_cat, col)
        for col in range(X_cat.shape[1])
    ])

    # -------------------------
    # zusammenführen
    # -------------------------
    #pz = np.concatenate([np.asarray(pz_num).ravel(),np.asarray(pz_cat).ravel()])

    return pz_num.astype(float), pz_cat.astype(str)
        
    