from src.explainer import get_top_risk_factors
from fastapi import FastAPI
from pydantic import BaseModel
from src.recommender import generate_recommendation
import pandas as pd
import joblib
import json

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

# Load model assets
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)


class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

    gender_Male: int = 0
    Partner_Yes: int = 0
    Dependents_Yes: int = 0
    PhoneService_Yes: int = 1

    MultipleLines_No_phone_service: int = 0
    MultipleLines_Yes: int = 0

    InternetService_Fiber_optic: int = 0
    InternetService_No: int = 0

    OnlineSecurity_No_internet_service: int = 0
    OnlineSecurity_Yes: int = 0

    OnlineBackup_No_internet_service: int = 0
    OnlineBackup_Yes: int = 0

    DeviceProtection_No_internet_service: int = 0
    DeviceProtection_Yes: int = 0

    TechSupport_No_internet_service: int = 0
    TechSupport_Yes: int = 0

    StreamingTV_No_internet_service: int = 0
    StreamingTV_Yes: int = 0

    StreamingMovies_No_internet_service: int = 0
    StreamingMovies_Yes: int = 0

    Contract_One_year: int = 0
    Contract_Two_year: int = 0

    PaperlessBilling_Yes: int = 0

    PaymentMethod_Credit_card_automatic: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaymentMethod_Mailed_check: int = 0


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: CustomerData):

    input_dict = {
        "SeniorCitizen": data.SeniorCitizen,
        "tenure": data.tenure,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges,

        "TenureChargeRatio":
            data.TotalCharges / (data.tenure + 1),

        "LongTermCustomer":
            int(data.tenure >= 24),

        "HighValueCustomer":
            int(data.MonthlyCharges >= 80),

        "gender_Male": data.gender_Male,
        "Partner_Yes": data.Partner_Yes,
        "Dependents_Yes": data.Dependents_Yes,
        "PhoneService_Yes": data.PhoneService_Yes,

        "MultipleLines_No phone service":
            data.MultipleLines_No_phone_service,

        "MultipleLines_Yes":
            data.MultipleLines_Yes,

        "InternetService_Fiber optic":
            data.InternetService_Fiber_optic,

        "InternetService_No":
            data.InternetService_No,

        "OnlineSecurity_No internet service":
            data.OnlineSecurity_No_internet_service,

        "OnlineSecurity_Yes":
            data.OnlineSecurity_Yes,

        "OnlineBackup_No internet service":
            data.OnlineBackup_No_internet_service,

        "OnlineBackup_Yes":
            data.OnlineBackup_Yes,

        "DeviceProtection_No internet service":
            data.DeviceProtection_No_internet_service,

        "DeviceProtection_Yes":
            data.DeviceProtection_Yes,

        "TechSupport_No internet service":
            data.TechSupport_No_internet_service,

        "TechSupport_Yes":
            data.TechSupport_Yes,

        "StreamingTV_No internet service":
            data.StreamingTV_No_internet_service,

        "StreamingTV_Yes":
            data.StreamingTV_Yes,

        "StreamingMovies_No internet service":
            data.StreamingMovies_No_internet_service,

        "StreamingMovies_Yes":
            data.StreamingMovies_Yes,

        "Contract_One year":
            data.Contract_One_year,

        "Contract_Two year":
            data.Contract_Two_year,

        "PaperlessBilling_Yes":
            data.PaperlessBilling_Yes,

        "PaymentMethod_Credit card (automatic)":
            data.PaymentMethod_Credit_card_automatic,

        "PaymentMethod_Electronic check":
            data.PaymentMethod_Electronic_check,

        "PaymentMethod_Mailed check":
            data.PaymentMethod_Mailed_check
    }

    input_df = pd.DataFrame(
        [input_dict]
    )

    input_df = input_df[
        feature_names
    ]

    input_scaled = scaler.transform(
        input_df
    )

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    prediction = (
        "High Risk"
        if probability >= 0.5
        else "Low Risk"
    )

    top_features = get_top_risk_factors()

    recommendations = generate_recommendation(
    input_dict
)

    return {

    "churn_probability":
        round(
            float(probability),
            4
        ),

    "prediction":
        prediction,

    "top_risk_factors":
        top_features,

    "recommendations":
        recommendations
}