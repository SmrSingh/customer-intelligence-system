import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load dataset
df = pd.read_csv("data/raw/telco_churn.csv")

print(df.head())
print(df.info())



df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)



df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)
print(df.isnull().sum())

df.drop("customerID", axis=1, inplace=True)

X = df.drop("Churn", axis=1)

y = df["Churn"]

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns

print(categorical_cols)

print(numerical_cols)
X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()

X_train[numerical_cols] = scaler.fit_transform(
    X_train[numerical_cols]
)

X_test[numerical_cols] = scaler.transform(
    X_test[numerical_cols]
)
X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)

pd.DataFrame(y_train).to_csv(
    "data/processed/y_train.csv",
    index=False
)

pd.DataFrame(y_test).to_csv(
    "data/processed/y_test.csv",
    index=False
)

print("Preprocessing completed successfully!")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)