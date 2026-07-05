# RetailPulse 🛍️

RetailPulse is an end-to-end **retail analytics and machine learning project** that helps businesses understand customer behavior, predict churn, forecast demand, and support inventory planning through an interactive **Streamlit dashboard**.

## Features
- **Customer Segmentation** using RFM + KMeans clustering
- **Churn Prediction** using XGBoost (**AUC: 0.81**)
- **Demand Forecasting** using Prophet with **weekly sales forecasting** (**MAPE: 13.0%**)
- **Inventory Optimization** using forecasted demand
- **Interactive Dashboard** built with Streamlit

## Tech Stack
- **Python, Pandas, NumPy**
- **Scikit-learn, XGBoost, Prophet**
- **SHAP, MLflow, Evidently**
- **Streamlit, Plotly, Matplotlib**

## Project Structure
```bash
Retailpulse/
├── data/
├── notebooks/
├── dashboard/
├── models/
├── reports/
├── src/
├── requirements.txt
└── README.md