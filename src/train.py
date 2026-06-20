import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load ML-ready dataset
df = pd.read_csv(
    "data/processed/ml_telco_churn.csv"
)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nDataset Shape:")
print(df.shape)

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:")
print(X_train.shape)

print("\nTest Shape:")
print(X_test.shape)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Applied")

# Apply SMOTE only on training data

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nSMOTE Applied")
print("Training Shape Before:", X_train.shape)
print("Training Shape After :", X_train_smote.shape)

# Logistic Regression Model
model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

model.fit(
    X_train_smote,
    y_train_smote
)
# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
print("\nModel Performance")
print("-" * 40)

print(
    f"Accuracy : {accuracy_score(y_test, y_pred):.4f}"
)

print(
    f"Precision: {precision_score(y_test, y_pred):.4f}"
)

print(
    f"Recall   : {recall_score(y_test, y_pred):.4f}"
)

print(
    f"F1 Score : {f1_score(y_test, y_pred):.4f}"
)

print(
    f"ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}"
)

# Save champion model

# Save XGBoost as champion model

joblib.dump(
    model,
    "models/churn_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# Save feature names

feature_names = list(X.columns)

with open(
    "models/feature_names.json",
    "w"
) as f:
    json.dump(
        feature_names,
        f,
        indent=4
    )

print("Feature Names Saved!")

print("\nModel Saved Successfully!")

print("\nConfusion Matrix")
print("-" * 40)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\n")
print("=" * 50)
print("RANDOM FOREST MODEL")
print("=" * 50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Performance")
print("-" * 40)

print(
    f"Accuracy : {accuracy_score(y_test, rf_pred):.4f}"
)

print(
    f"Precision: {precision_score(y_test, rf_pred):.4f}"
)

print(
    f"Recall   : {recall_score(y_test, rf_pred):.4f}"
)

print(
    f"F1 Score : {f1_score(y_test, rf_pred):.4f}"
)

print(
    f"ROC AUC  : {roc_auc_score(y_test, rf_prob):.4f}"
)
print("\n")
print("=" * 50)
print("XGBOOST MODEL")
print("=" * 50)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost Performance")
print("-" * 40)

print(
    f"Accuracy : {accuracy_score(y_test, xgb_pred):.4f}"
)

print(
    f"Precision: {precision_score(y_test, xgb_pred):.4f}"
)

print(
    f"Recall   : {recall_score(y_test, xgb_pred):.4f}"
)

print(
    f"F1 Score : {f1_score(y_test, xgb_pred):.4f}"
)

print(
    f"ROC AUC  : {roc_auc_score(y_test, xgb_prob):.4f}"
)