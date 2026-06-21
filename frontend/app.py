import pandas as pd
import os
import streamlit as st
import requests
from datetime import datetime

# ==========================
# Paths
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

os.makedirs(
    prediction_dir,
    exist_ok=True
)

history_file = os.path.join(
    prediction_dir,
    "prediction_history.csv"
)

# ==========================
# Streamlit Config
# ==========================

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
# Predict Button
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
            json=payload,
            timeout=30
        )

        result = response.json()

        history = {

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "tenure": tenure,

            "monthly_charges":
                monthly_charges,

            "total_charges":
                total_charges,

            "churn_probability":
                result["churn_probability"],

            "prediction":
                result["prediction"]
        }

        history_df = pd.DataFrame(
            [history]
        )

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

        st.subheader(
            "📈 Churn Risk Score"
        )

        st.metric(
            "Churn Probability",
            f"{probability:.2f}%"
        )

        st.progress(
            int(probability)
        )

        st.subheader(
            "⚠️ Risk Level"
        )

        if probability >= 70:

            st.error(
                result["prediction"]
            )

        elif probability >= 40:

            st.warning(
                result["prediction"]
            )

        else:

            st.success(
                result["prediction"]
            )

        st.subheader(
            "🔥 Top Risk Factors"
        )

        for factor in result[
            "top_risk_factors"
        ]:

            st.write(
                f"• {factor}"
            )

        st.subheader(
            "💡 Recommendations"
        )

        for rec in result[
            "recommendations"
        ]:

            st.success(rec)

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

# ==========================
# Prediction History
# ==========================

st.divider()

st.subheader(
    "📜 Prediction History"
)

if os.path.exists(history_file):

    history_df = pd.read_csv(
        history_file
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    # ==========================
    # Dashboard Metrics
    # ==========================

    st.subheader(
        "📊 Dashboard Metrics"
    )

    avg_churn = (
        history_df[
            "churn_probability"
        ].mean() * 100
    )

    total_predictions = len(
        history_df
    )

    high_risk_count = len(
        history_df[
            history_df[
                "prediction"
            ] == "High Risk"
        ]
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Predictions",
            total_predictions
        )

    with col2:

        st.metric(
            "High Risk Customers",
            high_risk_count
        )

    with col3:

        st.metric(
            "Average Churn %",
            f"{avg_churn:.2f}%"
        )

    # ==========================
    # Risk Distribution
    # ==========================

    st.subheader(
        "📊 Risk Distribution"
    )

    risk_counts = (
        history_df[
            "prediction"
        ].value_counts()
    )

    st.bar_chart(
        risk_counts
    )

    # ==========================
    # Churn Trend
    # ==========================

    st.subheader(
        "📈 Churn Probability Trend"
    )

    chart_df = history_df.copy()

    chart_df[
        "Prediction No"
    ] = range(
        1,
        len(chart_df) + 1
    )

    chart_df[
        "churn_probability"
    ] = (
        chart_df[
            "churn_probability"
        ] * 100
    )

    chart_df = chart_df.set_index(
        "Prediction No"
    )

    st.line_chart(
        chart_df[
            "churn_probability"
        ]
    )

    # ==========================
    # Recent Predictions
    # ==========================

    st.subheader(
        "🕒 Recent Predictions"
    )

    st.dataframe(
        history_df.tail(5),
        use_container_width=True
    )

    # ==========================
    # Download CSV
    # ==========================

    with open(
        history_file,
        "rb"
    ) as f:

        st.download_button(
            label=
            "⬇ Download Prediction History",
            data=f,
            file_name=
            "prediction_history.csv",
            mime="text/csv"
        )

    # ==========================
    # Clear History
    # ==========================

    if st.button(
        "🗑 Clear History"
    ):

        os.remove(
            history_file
        )

        st.success(
            "Prediction history cleared!"
        )

        st.rerun()

else:

    st.info(
        "No prediction history available yet."
    )

# ==========================
# Footer
# ==========================

st.divider()

st.markdown(
    """
    ### 🚀 Built By Preethi Beri

    **Technologies Used**

    - Python
    - Pandas
    - Scikit-Learn
    - FastAPI
    - Streamlit
    - Render Cloud

    **Customer Churn Prediction & Retention Recommendation Engine**
    """
)