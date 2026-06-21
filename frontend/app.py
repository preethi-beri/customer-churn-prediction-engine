import pandas as pd
import os
import streamlit as st
import requests

st.set_page_config(
    page_title="Customer Churn Prediction Engine",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Engine")

st.write(
    "Predict customer churn risk and generate personalized retention recommendations."
)

# ==========================
# Customer Inputs
# ==========================

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:

    tenure = st.slider(
        "Customer Tenure (Months)",
        0,
        72,
        12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        value=95.0
    )

    total_charges = st.number_input(
        "Total Charges",
        value=1140.0
    )

with col2:

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    contract = st.selectbox(
        "Contract Type",
        [
            "Month-to-Month",
            "One Year",
            "Two Year"
        ]
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "Fiber Optic",
            "DSL",
            "No Internet"
        ]
    )

# ==========================
# Prediction Button
# ==========================

if st.button("🚀 Predict Churn Risk"):

    contract_one_year = 0
    contract_two_year = 0

    if contract == "One Year":
        contract_one_year = 1

    elif contract == "Two Year":
        contract_two_year = 1

    fiber_optic = 0
    internet_no = 0

    if internet == "Fiber Optic":
        fiber_optic = 1

    elif internet == "No Internet":
        internet_no = 1

    payload = {

        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        "gender_Male": 1,
        "Partner_Yes": 0,
        "Dependents_Yes": 0,
        "PhoneService_Yes": 1,

        "MultipleLines_No_phone_service": 0,
        "MultipleLines_Yes": 1,

        "InternetService_Fiber_optic": fiber_optic,
        "InternetService_No": internet_no,

        "OnlineSecurity_No_internet_service": 0,
        "OnlineSecurity_Yes": 0,

        "OnlineBackup_No_internet_service": 0,
        "OnlineBackup_Yes": 0,

        "DeviceProtection_No_internet_service": 0,
        "DeviceProtection_Yes": 0,

        "TechSupport_No_internet_service": 0,
        "TechSupport_Yes": 0,

        "StreamingTV_No_internet_service": 0,
        "StreamingTV_Yes": 1,

        "StreamingMovies_No_internet_service": 0,
        "StreamingMovies_Yes": 1,

        "Contract_One_year": contract_one_year,
        "Contract_Two_year": contract_two_year,

        "PaperlessBilling_Yes": 1,

        "PaymentMethod_Credit_card_automatic": 0,
        "PaymentMethod_Electronic_check": 1,
        "PaymentMethod_Mailed_check": 0
    }

    try:

        response = requests.post(
            "https://customer-churn-api-cjiy.onrender.com/predict",
            json=payload
        )

        result = response.json()

        # ==========================
        # Save Prediction History
        # ==========================

        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        prediction_dir = os.path.join(
            BASE_DIR,
            "data",
            "predictions"
        )

        if not os.path.isdir(prediction_dir):
            os.makedirs(
                prediction_dir,
                exist_ok=True
            )

        history_file = os.path.join(
            prediction_dir,
            "prediction_history.csv"
        )

        history = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn_probability": result["churn_probability"],
            "prediction": result["prediction"]
        }

        history_df = pd.DataFrame([history])

        if os.path.exists(history_file):

            history_df.to_csv(
                history_file,
                mode="a",
                header=False,
                index=False
            )

        else:

            history_df.to_csv(
                history_file,
                index=False
            )

        probability = (
            result["churn_probability"] * 100
        )

        st.divider()

        # ==========================
        # Risk Score
        # ==========================

        st.subheader("📈 Churn Risk Score")

        st.metric(
            "Churn Probability",
            f"{probability:.2f}%"
        )

        st.progress(
            int(probability)
        )

        # ==========================
        # Risk Level
        # ==========================

        st.subheader("⚠️ Risk Level")

        if probability >= 70:
            st.error(result["prediction"])

        elif probability >= 40:
            st.warning(result["prediction"])

        else:
            st.success(result["prediction"])

        # ==========================
        # Top Risk Factors
        # ==========================

        st.subheader("🔥 Top Risk Factors")

        for factor in result["top_risk_factors"]:
            st.write(f"• {factor}")

        # ==========================
        # Recommendations
        # ==========================

        st.subheader("💡 Recommendations")

        for rec in result["recommendations"]:
            st.success(rec)

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

        # ==========================
# Prediction History
# ==========================

st.divider()

st.subheader("📜 Prediction History")

history_file = os.path.join(
    "data",
    "predictions",
    "prediction_history.csv"
)

if os.path.exists(history_file):

    history_df = pd.read_csv(
        history_file
    )

    st.dataframe(
        history_df.tail(10),
        use_container_width=True
    )

else:

    st.info(
        "No prediction history available yet."
    )