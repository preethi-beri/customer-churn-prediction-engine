# 📊 Customer Churn Prediction & Retention Recommendation Engine

An end-to-end Machine Learning application that predicts customer churn risk, identifies key risk factors, and generates personalized retention recommendations through an interactive dashboard.

## 🚀 Live Project

**API Endpoint:**
https://customer-churn-api-cjiy.onrender.com

## 🎯 Project Overview

Customer churn is one of the most critical business challenges in subscription-based industries. This project leverages Machine Learning to predict whether a customer is likely to churn and provides actionable recommendations to improve customer retention.

The solution includes:

* Churn Risk Prediction
* Customer Risk Classification
* Retention Recommendation Engine
* Explainable AI Risk Factors
* Prediction History Tracking
* Interactive Analytics Dashboard
* REST API Deployment

---

## 🏗️ System Architecture

Customer Data → Feature Engineering → Machine Learning Model → FastAPI API → Streamlit Dashboard → Retention Recommendations & Analytics

---

## ✨ Features

### 🔍 Churn Prediction

Predicts the probability of customer churn using a trained Machine Learning model.

### ⚠️ Risk Classification

Classifies customers into risk categories based on churn probability.

### 💡 Retention Recommendations

Generates personalized recommendations to improve customer retention.

### 📈 Analytics Dashboard

Provides:

* Churn Probability Visualization
* Prediction History Tracking
* Risk Distribution Analysis
* Trend Monitoring

### 📥 Report Export

Download prediction history as CSV reports.

---

## 🛠️ Technology Stack

### Programming Languages

* Python

### Machine Learning

* Scikit-Learn
* Logistic Regression
* Random Forest
* XGBoost
* SMOTE

### Data Processing

* Pandas
* NumPy

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Deployment

* Render

### Model Persistence

* Joblib

---

## 📂 Project Structure

```text
customer-churn-prediction-engine/
│
├── frontend/
│   └── app.py
│
├── src/
│   ├── api.py
│   ├── train.py
│   ├── recommender.py
│   └── explainer.py
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── feature_names.json
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
│
├── requirements.txt
└── README.md
```

---

## 📊 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 79.32% |
| Precision | 63.43% |
| Recall    | 52.41% |
| F1 Score  | 57.39% |
| ROC-AUC   | 83.75% |

---

## 📸 Dashboard Features

* Customer Churn Prediction
* Risk Level Detection
* Top Risk Factors
* Retention Recommendations
* Prediction History
* Risk Distribution Analytics
* Churn Trend Monitoring

---

## 🔮 Future Enhancements

* SHAP Explainable AI Visualizations
* Customer Segmentation
* Automated PDF Reports
* Docker Deployment
* Cloud Database Integration
* User Authentication

---

## 👩‍💻 Author

**Preethi Beri**

B.Tech – Computer Science & Engineering (Data Science)

GitHub: https://github.com/preethi-beri

LinkedIn: https://www.linkedin.com/in/preethi-beri/

---

## ⭐ If you found this project useful, consider giving it a star.
