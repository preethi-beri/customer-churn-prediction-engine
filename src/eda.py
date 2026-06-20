import pandas as pd

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_telco_churn.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(
    df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nContract vs Churn:")
print(
    pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ).round(3) * 100
)

print("\nInternet Service vs Churn:")
print(
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ).round(3) * 100
)
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder if needed
import os
os.makedirs("data/eda", exist_ok=True)

# Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.savefig("data/eda/churn_distribution.png")
plt.close()

# Contract vs Churn
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)
plt.title("Contract Type vs Churn")
plt.tight_layout()
plt.savefig("data/eda/contract_vs_churn.png")
plt.close()

# Internet Service vs Churn
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)
plt.title("Internet Service vs Churn")
plt.tight_layout()
plt.savefig("data/eda/internet_vs_churn.png")
plt.close()

print("\nEDA Charts Saved Successfully!")