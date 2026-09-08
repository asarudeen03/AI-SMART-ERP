# 📊 AI Smart ERP

An AI-powered Enterprise Resource Planning (ERP) Analytics System built using Python, Streamlit, Data Analytics, and Machine Learning.

This system helps businesses analyze sales performance, forecast revenue, predict product demand, and identify inventory reorder requirements through an interactive dashboard.

## 🚀 Features

### 📈 Dashboard Overview
- Overall revenue
- Total quantity sold
- Average monthly demand
- Total products
- Low-stock entries
- Best-selling product
- Monthly revenue trends
- Revenue by product

### 📊 Sales Analytics
- Product-wise revenue analysis
- Monthly revenue trends
- Quantity sold by product
- Product and category filtering

### 💰 Sales Prediction
Uses **Linear Regression** to predict expected revenue based on:
- Year
- Month
- Quantity Sold
- Unit Price

The module provides:
- R² Score
- Mean Absolute Error (MAE)
- Predicted Revenue

### 📦 Demand Prediction
Uses **Random Forest Regressor** to predict monthly product demand using:
- Product
- Category
- Stock Available
- Reorder Level
- Quantity Sold

The system also identifies potential stock shortages and recommends reorder quantities.

### 🔄 Smart Reorder Insights
Identifies products that may require stock replenishment by comparing predicted demand with available stock.

### 💡 Business Insights
Provides:
- Top-selling product
- Slow-moving product
- Highest-revenue product
- Top 5 best-selling products
- Bottom 5 slow-moving products

## 🧠 Machine Learning Models

| Module | Model | Purpose |
|---|---|---|
| Sales Prediction | Linear Regression | Revenue prediction |
| Demand Prediction | Random Forest Regressor | Monthly demand prediction |

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## 📁 Project Structure

```text
AI-SMART-ERP/
│
├── analytics/
│   └── data_analysis.py
│
├── data/
│   └── Smart_ERP_Final_Dataset.csv
│
├── ml/
│   ├── demand_prediction.py
│   └── sales_prediction.py
│
├── app.py
└── .gitignore
