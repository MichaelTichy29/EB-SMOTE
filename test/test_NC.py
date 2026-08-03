import pandas as pd
import numpy as np
#from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import openpyxl
from sklearn.neighbors import NearestNeighbors


from enh_border_NC import generate_dataset_NC
from enh_border_level_NC import generate_dataset_level_NC
from enh_border_level_NC import EnhancedBorderlineNC


df_org = pd.read_csv('adult_clean_imb.csv', encoding = 'cp850')

# 5 % der Zeilen
df_5 = df_org.sample(frac=0.10, random_state=42)



# preprocess
# "fnlwgt" ist kein Personenmerkmal
#  "education" ist redundant zu education num.

df_5.drop(columns=["fnlwgt", "education"], inplace=True)
df_5["native_country"] = (df_5["native_country"] == "United-States").astype(float)
#df_5["native_country"] = np.where(df_5["native_country"] == "United-States", 1.0, 0.0)

continuous_cols = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country"
]

categorical_cols = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex"
]



X = df_5.drop(columns=["Class"])
y = df_5["Class"]
#native country? drop? or us/non us?

#X = df.drop(columns=["target"])

cont_idx = [X.columns.get_loc(col) for col in continuous_cols]
cat_idx  = [X.columns.get_loc(col) for col in categorical_cols]

#X_values = X.to_numpy(dtype=object) in den einzelnen Aufrufen in gen_Dataset
#y_values = np.asarray(y) in generate_Dataset




X_train, X_test, y_train, y_test = train_test_split(
     X, y,
     test_size=0.2,      # 20% Test
     train_size=0.8,     # 80% Train
     stratify=y)


# in generate Dataset
#X_num = X_train[continuous_cols].to_numpy(dtype=float)
#X_cat = X_train[categorical_cols].astype(str).to_numpy()


   
#########################################
# X_train + y_train zusammenführen
train_array = np.column_stack((X_train, y_train))

# DataFrame daraus machen
train_df = pd.DataFrame(train_array)

# Letzte Spalte benennen
train_df.rename(columns={train_df.columns[-1]: "target"}, inplace=True)

# Export
train_df.to_excel("train_data_vor.xlsx", index=False)

###

## Hierauf kommt es mir an
#sampler = RandomOverSampler()
#X_train, y_train  = sampler.fit_resample(X_train, y_train)
## Hierauf kommt es mir an
#sampler = RandomOverSampler()
#X_train, y_train  = sampler.fit_resample(X_train, y_train)

# em oder cm
"""
scaler = StandardScaler()
X_num = scaler.fit_transform(X[continuous_cols])

X_cat = X[categorical_cols].astype(str).to_numpy()
"""

aus = 1
if aus == 1: 
    samp_meth = "cm" 
    level = 0
    balancing = 1.0
    
    sampler = EnhancedBorderlineNC(
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        minority_label=1,
        k1=3,
        k2=10,
        samp_meth="cm",
        random_state=42,
        balance_level=1.0)
    
    
    X_train, y_train = sampler.fit_resample(X_train, y_train)
    """
    if level == 1:
        X_train, y_train = generate_dataset_NC(X_train, y_train, continuous_cols, categorical_cols,1, 3, 10, samp_meth, 42)
    else:
        X_train, y_train = generate_dataset_level_NC(X_train, y_train, continuous_cols, categorical_cols, 1, 3, 10, samp_meth, 42, balancing)
    """
    
    
    ###
    """
    # X_train + y_train zusammenführen
    train_array = np.column_stack((X_train, y_train))
    
    # DataFrame daraus machen
    train_df = pd.DataFrame(train_array)
    
    # Letzte Spalte benennen
    train_df.rename(columns={train_df.columns[-1]: "target"}, inplace=True)
    
    # Export
    train_df.to_excel("train_data_nach.xlsx", index=False)
    
    """
    cols = continuous_cols + categorical_cols
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test  = pd.DataFrame(X_test, columns=cols)
    
    
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    
    
    #############################################
    an = 1
    if an == 1:
        # Modell
        model = DecisionTreeClassifier(random_state=42)
        
        # Trainieren
        model.fit(X_train, y_train)
        
        # Vorhersage
        y_pred = model.predict(X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(cm)
        
        # Schön darstellen
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        
        
        plt.show()
        
    
