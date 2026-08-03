import numpy as np

from sklearn.neighbors import NearestNeighbors
from pointdef import generate_sample_cm, generate_sample_em



def generate_dataset(X, y, minority_label=1, k1=3, k2=10, samp_meth="em"): #, random_state=42):

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

    synthetic_samples = []

    for i, global_idx in enumerate(minority_idx):

        p = X_values[global_idx]

        # zufälliger minority Nachbar
        chosen_local = rng.choice(ind_min[i])
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

        # dein Regelwerk
        if samp_meth == "em":
            pz = generate_sample_em(p, pj, mp, mj, nn_all, X_values, rng)
        elif samp_meth == "cm":
            pz = generate_sample_cm(p, pj, mp, mj, nn_all, X_values, rng)
                           
        if pz is not None:
            synthetic_samples.append(pz)

    # Ergebnis
    if len(synthetic_samples) == 0:
        return X, y

    X_syn = np.array(synthetic_samples)
    y_syn = np.full(len(X_syn), minority_label)

    X_res = np.vstack([X_values, X_syn])
    y_res = np.hstack([y_values, y_syn])

    return X_res, y_res