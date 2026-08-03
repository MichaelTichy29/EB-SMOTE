import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import openpyxl

from enh_border import generate_dataset
from enh_border_level import generate_dataset_level


df_org = pd.read_csv('creditcard.csv', encoding = 'cp850')

# 5 % der Zeilen
df_5 = df_org.head(int(len(df_org) * 0.03))

X = df_5.drop(columns=["Class"])
y = df_5["Class"]

#test = df_5["Class"].value_counts()
#print("test = ", test)

# preprocess
mask = (X != 0).all(axis=1)
X = X[mask]
y = y[mask]

scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% Test
    train_size=0.8,     # 80% Train
    stratify=y
)
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

samp_meth = "em" # oder cm
level = 0
if level == 0:
    X_train, y_train = generate_dataset(X_train, y_train, 1, 3, 10, samp_meth, 42)
else:
    X_train, y_train = generate_dataset_level(X_train, y_train, 1, 3, 10, samp_meth, 42, 1.0)

###

# X_train + y_train zusammenführen
train_array = np.column_stack((X_train, y_train))

# DataFrame daraus machen
train_df = pd.DataFrame(train_array)

# Letzte Spalte benennen
train_df.rename(columns={train_df.columns[-1]: "target"}, inplace=True)

# Export
train_df.to_excel("train_data_nach.xlsx", index=False)




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
