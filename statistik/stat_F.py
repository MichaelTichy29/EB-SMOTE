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
from collections import Counter

from scipy.stats import f

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
max_depth=10
#random_state=1
C = 1.0
#model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
model = LogisticRegression(C=1.0)


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
do_sl_smote =0
k_neighbors = min(k1, min_class - 1)
#sampler2 = sv.Safe_Level_SMOTE(k_neighbors=k_neighbors,sampling_strategy=balance_level)
#

###############
# sampler 1: EB
k1=5
k2=10
samp_meth= "em"
sampler1 = EnhancedBorderline(k1=k1, k2=k2, samp_meth= samp_meth, balance_level=balance_level) #, random_state=random_state)        
#
#samp_meth2= "cm"
#sampler2 = EnhancedBorderline(k1=k1, k2=k2, samp_meth=samp_meth2, balance_level=balance_level) #, random_state=random_state)        





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
        #random_state=random_state
    )

    # 1. Durchlauf:
    # train -> test
        
    X_res, y_res = sampler1.fit_resample(X_train, y_train)
    
    # Modell auf den resampleten Trainingsdaten trainieren
    model.fit(X_res, y_res)
    
    # Vorhersage auf den unveränderten Testdaten
    y_pred = model.predict(X_test)
    
    # Precision und Recall berechnen
    prec_k1A = precision_score(y_test, y_pred, zero_division=0)
    
    rec_k1A = recall_score(y_test, y_pred,zero_division=0)
        
    
    
    ####
    
    
    if do_sl_smote == 1:
        feature_names = (
            X_train.columns
            if isinstance(X_train, pd.DataFrame)
            else None
            )        
    
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
    y_pred = model.predict(X_test)
    
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
    
    # Modell auf den resampleten Trainingsdaten trainieren
    model.fit(X_res, y_res)
    
    # Vorhersage auf den unveränderten Testdaten
    y_pred = model.predict(X_test)
    
    # Precision und Recall berechnen
    prec_k2A = precision_score(y_test, y_pred, zero_division=0)
    
    rec_k2A = recall_score(y_test, y_pred,zero_division=0)
    
    ####
    
    
    if do_sl_smote == 1:
        feature_names = (
            X_train.columns
            if isinstance(X_train, pd.DataFrame)
            else None
            )        
    
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
    y_pred = model.predict(X_test)
    
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
    print("rec_k = ", rec_k)
    
    
    
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

  

