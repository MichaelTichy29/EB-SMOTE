import pandas as pd

cols = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "class"
]

# 1. laden
df = pd.read_csv(
    "adult.data",
    names=cols,
    skipinitialspace=True
)

# 2. Missing Values markieren
df.replace("?", pd.NA, inplace=True)

# 3. Missing Values entfernen
df.dropna(inplace=True)


# 3.5 umkodieren
df["class"] = df["class"].str.strip()

df["class"] = df["class"].map({
    "<=50K": 0,
    ">50K": 1
})


# 3.55
df.drop(columns=["fnlwgt", "education"], inplace=True)
df["native_country"] = (df["native_country"] == "United-States").astype(float)


#3.6 umbennenen
df = df.rename(columns={"class": "Class"})


#print("erstens", df["class"].value_counts().sort_index())

# 4. speichern
df.to_csv("adult_clean.csv", index=False)


# 5. Imbalance

df_min = df[df["Class"] == 1].sample(frac=0.2, random_state=42)
df_maj = df[df["Class"] == 0]

df_imb = pd.concat([df_maj, df_min])


# 6. speichern
df_imb.to_csv("adult_clean_imb.csv", index=False)

#print("2.", df_imb["class"].value_counts().sort_index())

