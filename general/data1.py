
####################################
#####     Preprocessing     #######
####################################


from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess(X, y, config):

    # ---------------------------------
    # Nullzeilen entfernen
    # ---------------------------------
    if config["preprocess"]["drop_zero_rows"]:

        mask = (X != 0).all(axis=1)

        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

    # ---------------------------------
    # Normalisierung
    # ---------------------------------
    if config["preprocess"]["normalize"]:

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # zurück zu DataFrame
        X = pd.DataFrame(
            X_scaled,
            columns=X.columns
        )

    return X, y


def preprocess_NC(X, y, config):

    if config["preprocess"]["normalize"]:

        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y
