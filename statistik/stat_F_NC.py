from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC
from enh_border_level_NC import EnhancedBorderlineNC
from sklearn.metrics import precision_score, recall_score

from scipy.stats import f

def preprocess_NC(X, y):

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y




df_org = pd.read_csv("adult_clean_imb.csv", sep=",", encoding="utf-8")


# Optional Sampling
df = df_org.sample(frac=0.2, random_state=1)

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
balance_level = 0.6
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
k2=15
samp_meth= "cm"
continuous_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
categorical_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex']
sampler1 = EnhancedBorderlineNC(continuous_cols=continuous_cols, categorical_cols=categorical_cols, k1=k1, k2=k2, samp_meth= samp_meth, balance_level=balance_level, random_state=random_state)        




prec_1 = []
prec_2 = []
s_prec = []
rec_1 = []
rec_2 = []
s_rec = []

for i in range(5):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        stratify=y,
        random_state=random_state
    )

    # 1. Durchlauf:
    # train -> test
        
    X_res, y_res = sampler1.fit_resample(X_train, y_train)
    
    ##
    X_train_enc = pd.get_dummies(X_res)
    X_test_enc = pd.get_dummies(X_test)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    ##
    
    # Modell auf den resampleten Trainingsdaten trainieren
    model.fit(X_train_enc, y_res)
    
    # Vorhersage auf den unveränderten Testdaten
    y_pred = model.predict(X_test_enc)
    
    # Precision und Recall berechnen
    prec_k1A = precision_score(y_test, y_pred, zero_division=0)
    
    rec_k1A = recall_score(y_test, y_pred,zero_division=0)
        
    
    ####
    
    
    X_res, y_res = sampler2.fit_resample(X_train, y_train)
    
    ##
    X_train_enc = pd.get_dummies(X_res)
    X_test_enc = pd.get_dummies(X_test)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    ##

    
    # Modell auf den resampleten Trainingsdaten trainieren
    model.fit(X_train_enc, y_res)
    
    # Vorhersage auf den unveränderten Testdaten
    y_pred = model.predict(X_test_enc)
    
    # Precision und Recall berechnen
    prec_k1B = precision_score(y_test, y_pred, zero_division=0)
    
    rec_k1B = recall_score(y_test, y_pred,zero_division=0)
    
    
    ####
    prec_k1 = prec_k1A - prec_k1B
    rec_k1 = rec_k1A - rec_k1B
    

    #####################################################################################
    # 2. Durchlauf:
    # Rollen tauschen
    X_train2 = X_test
    X_test2 = X_train
    y_train2 = y_test
    y_test2 = y_train
    #
    X_train = X_train2
    X_test = X_test2
    y_train = y_train2
    y_test = y_test2

    X_res, y_res = sampler1.fit_resample(X_train, y_train)
    
    ##
    X_train_enc = pd.get_dummies(X_res)
    X_test_enc = pd.get_dummies(X_test)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    ##
    
    # Modell auf den resampleten Trainingsdaten trainieren
    model.fit(X_train_enc, y_res)
    
    # Vorhersage auf den unveränderten Testdaten
    y_pred = model.predict(X_test_enc)
    
    # Precision und Recall berechnen
    prec_k2A = precision_score(y_test, y_pred, zero_division=0)
    
    rec_k2A = recall_score(y_test, y_pred,zero_division=0)
    
    ####
    
    
    X_res, y_res = sampler2.fit_resample(X_train, y_train)
    
    
    
    ##
    X_train_enc = pd.get_dummies(X_res)
    X_test_enc = pd.get_dummies(X_test)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    ##

    
    # Modell auf den resampleten Trainingsdaten trainieren
    model.fit(X_train_enc, y_res)
    
    # Vorhersage auf den unveränderten Testdaten
    y_pred = model.predict(X_test_enc)
    
    # Precision und Recall berechnen
    prec_k2B = precision_score(y_test, y_pred, zero_division=0)
    
    rec_k2B = recall_score(y_test, y_pred,zero_division=0)
    
    ####
    prec_k2 = prec_k2A - prec_k2B
    rec_k2 = rec_k2A - rec_k2B
    
    ####

    prec_k = (prec_k1 + prec_k2)/2
    s_prec_k =  (prec_k1 - prec_k)**2 + (prec_k2 - prec_k)**2 
    #
    rec_k = (rec_k1 + rec_k2)/2
    s_rec_k =  (rec_k1 - rec_k)**2 + (rec_k2 - rec_k)**2 
    #
    
    prec_1.append(prec_k1)
    prec_2.append(prec_k2)
    s_prec.append(s_prec_k)
    
    rec_1.append(rec_k1)
    rec_2.append(rec_k2)
    s_rec.append(s_rec_k)
    print("prec_k = ", prec_k)
    print("s_rec = ", s_rec)
    
    
    
sum_quad_prec = sum(d**2 for d in prec_1) + sum(d**2 for d in prec_2)
sum_sk_prec = sum(s_prec)
sum_quad_rec = sum(d**2 for d in rec_1) + sum(d**2 for d in rec_2)
sum_sk_rec = sum(s_rec)



T_prec = sum_quad_prec/(2*sum_sk_prec)
T_rec = sum_quad_rec/(2*sum_sk_rec)

print("T_prec = ", T_prec)
print("T_rec = ", T_rec)


p_value_prec = 1 - f.cdf(T_prec, dfn=10, dfd=5)
print("p value prec = ", p_value_prec)

p_value_rec = 1 - f.cdf(T_rec, dfn=10, dfd=5)
print("p value rrec = ", p_value_rec)

alpha = 0.05
F_crit = f.ppf(1 - alpha, dfn=10, dfd=5)
print(F_crit)

  

