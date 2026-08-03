


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


def generate_sample_cm(p, pj, m_p, m_j, nn_all, X_values, rng):
    
    # -------------------------------------------------
    # Regel 1
    # -------------------------------------------------
    if m_p == 0 and m_j == 0:
        return None

    # -------------------------------------------------
    # Regel 2
    # -------------------------------------------------
    elif m_p == 0 and m_j > 0:

        _, idx = nn_all.kneighbors(pj.reshape(1, -1))
        idx = idx[0][1]                 # erster echter Nachbar
        x_z = X_values[idx]

        r = rng.uniform(0.0, 0.5)

        p_z = pj + r * (x_z - pj)
        return p_z

    # -------------------------------------------------
    # Regel 3
    # -------------------------------------------------
    elif m_p > 0 and m_j == 0:

        _, idx = nn_all.kneighbors(p.reshape(1, -1))
        idx = idx[0][1]
        x_z = X_values[idx]

        r = rng.uniform(0.0, 0.5)

        p_z = p + r * (x_z - p)
        return p_z

    # -------------------------------------------------
    # Regel 4
    # -------------------------------------------------
    elif m_p == m_j:

        r = rng.uniform(0.0, 1.0)

        p_z = p + r * (pj - p)
        return p_z

    # -------------------------------------------------
    # Regel 5
    # -------------------------------------------------
    elif m_p < m_j:

        t = m_j / (m_p + m_j)
        r = rng.uniform(t, 1.0)

        p_z = p + r * (pj - p)
        return p_z

    # -------------------------------------------------
    # Regel 6
    # -------------------------------------------------
    else:   # m_j < m_p

        t = m_j / (m_p + m_j)
        r = rng.uniform(0.0, t)

        p_z = p + r * (pj - p)
        return p_z



def generate_sample_em(p, pj, m_p, m_j, nn_all, X_values, rng):
    
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

        p_z = p + r * (pj - p)
        return p_z

    # -------------------------------------------------
    # Regel 5
    # -------------------------------------------------
    elif m_p < m_j:

        t = m_p / (m_p + m_j)
        r = rng.uniform(0.0, t)

        p_z = p + r * (pj - p)
        return p_z

    # -------------------------------------------------
    # Regel 6
    # -------------------------------------------------
    else:   # m_j < m_p

        t = m_p / (m_p + m_j)
        r = rng.uniform(t, 1.0)

        p_z = p + r * (pj - p)
        return p_z
