from imblearn.over_sampling import BorderlineSMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE
from imblearn.over_sampling import ADASYN
import smote_variants as sv
from enh_border_level import EnhancedBorderline
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from scipy.stats import binom, chi2
from collections import Counter


def preprocess(X, y):

    # ---------------------------------
    # Nullzeilen entfernen
    # ---------------------------------

    mask = (X != 0).all(axis=1)

    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # ---------------------------------
    # Normalisierung
    # ---------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # zurück zu DataFrame
    X = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    return X, y


df_org = pd.read_csv("creditcard.csv", sep=",", encoding="utf-8")


# Optional Sampling
df = df_org.sample(frac=0.25, random_state=1)

# ---------------------------------
# pandas behalten!
# ---------------------------------
# Features und Target trennen
X = df.drop(columns=["Class"])
y = df["Class"]

# ---------------------------------
# Preprocessing
# ---------------------------------
X, y = preprocess(X, y)


#Gemeinsame Parameter
balance_level = 1.0
#classifier
#max_depth=10
random_state=1
C = 1
#model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
model = LogisticRegression(C=1.0, random_state=random_state)

# sampler 2: test
#-- Random
k1=5
#sampler2 = RandomOverSampler(sampling_strategy=balance_level)
#
#--SMOTE
#sampler2 = SMOTE(k_neighbors= k1,sampling_strategy=balance_level)
#
#-- ADAYSN
class_counts = Counter(y)
min_class = min(class_counts.values())
n_neighbors = min(k1, min_class - 1)
#sampler2 = ADASYN(n_neighbors=n_neighbors,sampling_strategy=balance_level)
#
#-- Borderline
k_neighbors = min(k1, min_class - 1)
sampler2 = BorderlineSMOTE(kind="borderline-2", k_neighbors=k_neighbors,sampling_strategy=balance_level)
#
#-- SL Smote
do_sl_smote =1
k_neighbors = min(k1, min_class - 1)
#sampler2 = sv.Safe_Level_SMOTE(k_neighbors=k_neighbors,sampling_strategy=balance_level)
#

###############
# sampler 1: EB
k1=5
k2=10
samp_meth= "cm"
sampler1 = EnhancedBorderline(k1=k1, k2=k2, samp_meth= samp_meth, balance_level=balance_level, random_state=random_state)        





X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=1
    )

if do_sl_smote == 1:
    feature_names = (
        X_train.columns
        if isinstance(X_train, pd.DataFrame)
        else None
        )        


X_res, y_res = sampler1.fit_resample(X_train, y_train)


# Modell auf den resampleten Trainingsdaten trainieren
model.fit(X_res, y_res)

# Vorhersage auf den unveränderten Testdaten
y_pred_a = model.predict(X_test)
    
tna, fpa, fna, tpa = confusion_matrix(y_test, y_pred_a).ravel()
cm_a = confusion_matrix(y_test, y_pred_a)

print("Konfusion matrix zu a ist:", cm_a)

######   
if do_sl_smote == 1:
    X_in = (
        X_train.to_numpy()
        if isinstance(X_train, pd.DataFrame)
        else np.asarray(X_train)
        )
    
    y_in = (
        y_train.to_numpy()
        if isinstance(y_train, pd.Series)
        else np.asarray(y_train)
    )
    
    X_res, y_res = sampler2.fit_resample(X_in, y_in)
    if feature_names is not None:
        X_res = pd.DataFrame(X_res, columns=feature_names)
    else:
        X_res = pd.DataFrame(X_res)
    
    y_res = pd.Series(y_res, name=getattr(y_train, "name", "Class"))
else: 
    X_res, y_res = sampler2.fit_resample(X_train, y_train)






# Modell auf den resampleten Trainingsdaten trainieren
model.fit(X_res, y_res)

# Vorhersage auf den unveränderten Testdaten
y_pred_b = model.predict(X_test)
    
tnb, fpb, fnb, tpb = confusion_matrix(y_test, y_pred_b).ravel()
cm_b = confusion_matrix(y_test, y_pred_b)

print("Konfusion matrix zu b ist:", cm_b)

a = ((y_pred_a == y_test) & (y_pred_b == y_test)).sum()
b = ((y_pred_a == y_test) & (y_pred_b != y_test)).sum()
c = ((y_pred_a != y_test) & (y_pred_b == y_test)).sum()
d = ((y_pred_a != y_test) & (y_pred_b != y_test)).sum()

table = [[int(a), int(b)], [int(c), int(d)]]
print("Pseudoconfusion = ", table)

T = (abs(b-c)-1)**2/(b+c)


p = chi2.sf(T, df=1)

print("p wert approx mc nemar = ", p)


###

n = b + c
x = min(b, c)

p = min(1.0, 2 * binom.cdf(x, n, 0.5))

print("Exakter p WErt ueber binom ist", p)


