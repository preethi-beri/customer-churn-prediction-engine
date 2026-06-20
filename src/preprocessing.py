import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/telco_churn.csv")

print("\nOriginal Shape:")
print(df.shape)

# Convert blank spaces to NaN
df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)

# Convert to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing Values After Conversion:")
print(df["TotalCharges"].isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna()
# Feature Engineering

# Average spending across customer lifetime
df["TenureChargeRatio"] = (
    df["TotalCharges"] / (df["tenure"] + 1)
)

# Customer has stayed 24+ months
df["LongTermCustomer"] = (
    df["tenure"] >= 24
).astype(int)

# Monthly bill above threshold
df["HighValueCustomer"] = (
    df["MonthlyCharges"] >= 80
).astype(int)

print("\nNew Features Created:")
print([
    "TenureChargeRatio",
    "LongTermCustomer",
    "HighValueCustomer"
])

print(
    df[
        [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "TenureChargeRatio",
            "LongTermCustomer",
            "HighValueCustomer"
        ]
    ].head()
)

print("\nShape After Cleaning:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

# -----------------------------
# ML Dataset Preparation
# -----------------------------

# Remove customer identifier
df_ml = df.drop("customerID", axis=1)

# Convert target variable
df_ml["Churn"] = df_ml["Churn"].map({
    "No": 0,
    "Yes": 1
})

# One-hot encode categorical features
df_encoded = pd.get_dummies(
    df_ml,
    drop_first=True
)

print("\nEncoded Dataset Shape:")
print(df_encoded.shape)

# Save ML-ready dataset
df_encoded.to_csv(
    "data/processed/ml_telco_churn.csv",
    index=False
)

print("\nML-ready dataset saved!")

# Save cleaned dataset
df.to_csv(
    "data/processed/cleaned_telco_churn.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")