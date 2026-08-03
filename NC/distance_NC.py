import numpy as np

def compute_median_std(df, continuous_cols):
    stds = df[continuous_cols].std(ddof=1)
    return np.median(stds)

def compute_median_std_num(X_num):
    stds = np.std(X_num, axis=0, ddof=1)
    return np.median(stds)



def dist(i, j, X_num, X_cat, M):
    d_cont_sq = np.sum((X_num[i] - X_num[j]) ** 2)
    n_diff = np.sum(X_cat[i] != X_cat[j])
    return d_cont_sq + (M * n_diff) ** 2

# =====================================================
# k nächste Nachbarn im gesamten Datensatz
# =====================================================
def knn_global(i, k, X_num, X_cat, M):
    
    n = len(X_num)
    
    dist_list = []
    
    for j in range(n):
        if j == i:
            continue
        
        d = dist(i, j, X_num, X_cat, M)
        dist_list.append((j, d))
    
    dist_list.sort(key=lambda x: x[1])
    
    return [idx for idx, _ in dist_list[:k]]


# =====================================================
# k nächste Nachbarn nur innerhalb Minority
# =====================================================
def knn_minority(i, k, minority_idx, X_num, X_cat, M):
    
    dist_list = []
    
    for j in minority_idx:
        if j == i:
            continue
        
        d = dist(i, j, X_num, X_cat, M)
        dist_list.append((j, d))
    
    dist_list.sort(key=lambda x: x[1])
    
    return [idx for idx, _ in dist_list[:k]]
