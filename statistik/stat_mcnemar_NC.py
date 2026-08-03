from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC
from enh_border_level_NC import EnhancedBorderlineNC
from sklearn.metrics import confusion_matrix
from scipy.stats import binom, chi2



def preprocess_NC(X, y):

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y




df_org = pd.read_csv("adult_clean_imb.csv", sep=",", encoding="utf-8")


# Optional Sampling
df = df_org.sample(frac=0.05, random_state=1)

# ---------------------------------
# pandas behalten!
# ---------------------------------
# Features und Target trennen
X = df.drop(columns=["Class"])
y = df["Class"]

# ---------------------------------
# Preprocessing
# ---------------------------------
X, y = preprocess_NC(X, y)


#Gemeinsame Parameter
balance_level = 1.0
#classifier
max_depth=10
random_state=1
model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)


# sampler 2: 
#-- Random    
k1=5
sampler2 = RandomOverSampler(sampling_strategy=balance_level)
#
#--SMOTENC
categorical_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex']
for col in categorical_cols:
    X[col] = X[col].astype("category")
cat_idx = [X.columns.get_loc(c) for c in categorical_cols]
#sampler2 = SMOTENC(categorical_features=cat_idx, k_neighbors= k1,sampling_strategy=balance_level)
#
###############
# sampler 1: EB
k1=5
k2=12
samp_meth= "cm"
continuous_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
categorical_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex']
sampler1 = EnhancedBorderlineNC(continuous_cols=continuous_cols, categorical_cols=categorical_cols, k1=k1, k2=k2, samp_meth= samp_meth, balance_level=balance_level, random_state=random_state)        






X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=1
    )

X_res, y_res = sampler1.fit_resample(X_train, y_train)


##
X_train_enc = pd.get_dummies(X_res)
X_test_enc = pd.get_dummies(X_test)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
##



# Modell auf den resampleten Trainingsdaten trainieren
model.fit(X_train_enc, y_res)



# Vorhersage auf den unveränderten Testdaten
y_pred_a = model.predict(X_test_enc)
    
tna, fpa, fna, tpa = confusion_matrix(y_test, y_pred_a).ravel()
cm_a = confusion_matrix(y_test, y_pred_a)

print("Konfusion matrix zu a ist:", cm_a)

######   
X_res, y_res = sampler2.fit_resample(X_train, y_train)


##
X_train_enc = pd.get_dummies(X_res)
X_test_enc = pd.get_dummies(X_test)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
##



# Modell auf den resampleten Trainingsdaten trainieren
model.fit(X_train_enc, y_res)

# Vorhersage auf den unveränderten Testdaten
y_pred_b = model.predict(X_test_enc)
    
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


