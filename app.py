import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Smart ERP System",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: linear-gradient(180deg, #020617 0%, #061127 100%);
    color: white;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.sidebar-title {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.sidebar-sub {
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.5;
    margin-bottom: 1rem;
}

.hero-box {
    background: linear-gradient(135deg, rgba(59,130,246,0.20), rgba(139,92,246,0.15));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 26px 28px;
    margin-bottom: 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.20);
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 16px;
    color: #cbd5e1;
    line-height: 1.7;
}

.section-title {
    font-size: 30px;
    font-weight: 800;
    color: white;
    margin-top: 8px;
    margin-bottom: 18px;
}

.subtle-text {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 14px;
}

.card {
    background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(30,41,59,0.95));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 22px;
    min-height: 135px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.18);
}

.card-label {
    font-size: 15px;
    color: #cbd5e1;
    margin-bottom: 18px;
    font-weight: 600;
}

.card-value {
    font-size: 26px;
    font-weight: 800;
    color: white;
}

.panel {
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.90));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 22px;
    margin-top: 8px;
    margin-bottom: 18px;
}

.panel-title {
    font-size: 20px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.small-note {
    color: #94a3b8;
    font-size: 14px;
}

.result-good {
    background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(5,150,105,0.15));
    border: 1px solid rgba(16,185,129,0.35);
    color: #d1fae5;
    padding: 16px 18px;
    border-radius: 16px;
    font-size: 18px;
    font-weight: 700;
    margin-top: 12px;
}

.result-warn {
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(127,29,29,0.15));
    border: 1px solid rgba(239,68,68,0.35);
    color: #fecaca;
    padding: 16px 18px;
    border-radius: 16px;
    font-size: 18px;
    font-weight: 700;
    margin-top: 12px;
}

.result-info {
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(30,64,175,0.15));
    border: 1px solid rgba(59,130,246,0.35);
    color: #dbeafe;
    padding: 16px 18px;
    border-radius: 16px;
    font-size: 17px;
    font-weight: 600;
    margin-top: 12px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(30,41,59,0.95));
    border: 1px solid rgba(255,255,255,0.08);
    padding: 16px;
    border-radius: 18px;
}

div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.12);
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-weight: 700;
    padding: 0.7rem 1rem;
}

.stButton > button:hover {
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
}

hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("data/Smart_ERP_Final_Dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown('<div class="sidebar-title">📦 Smart ERP</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-sub">AI-powered business analytics, forecasting, and inventory intelligence dashboard.</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Dashboard Overview",
        "Sales Analytics",
        "Sales Prediction",
        "Demand Prediction",
        "Smart Reorder Insights",
        "Business Insights",
        "About Project"
    ]
)

# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">📊 AI-Based Smart ERP Analytics System</div>
    <div class="hero-subtitle">
        A smart business dashboard built using Machine Learning and Data Analytics to monitor
        sales performance, forecast revenue, predict future demand, and support better inventory decisions.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD OVERVIEW
# --------------------------------------------------
if menu == "Dashboard Overview":
    st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle-text">A complete summary of business performance from the ERP dataset.</div>', unsafe_allow_html=True)

    total_revenue = df["Revenue"].sum()
    total_quantity = df["Quantity_Sold"].sum()
    avg_demand = df["Monthly_Demand"].mean()
    total_products = df["Product"].nunique()
    low_stock_count = (df["Stock_Available"] < df["Reorder_Level"]).sum()
    best_selling_product = df.groupby("Product")["Quantity_Sold"].sum().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="card"><div class="card-label">Overall Revenue</div><div class="card-value">₹ {total_revenue:,.0f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="card-label">Total Quantity Sold</div><div class="card-value">{total_quantity:,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="card-label">Average Monthly Demand</div><div class="card-value">{avg_demand:.1f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card"><div class="card-label">Total Products</div><div class="card-value">{total_products}</div></div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f'<div class="card"><div class="card-label">Low Stock Entries</div><div class="card-value">{low_stock_count}</div></div>', unsafe_allow_html=True)
    with c6:
        st.markdown(f'<div class="card"><div class="card-label">Best Selling Product</div><div class="card-value">{best_selling_product}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Revenue Trend</div>', unsafe_allow_html=True)
    monthly_revenue = df.groupby("Month")["Revenue"].sum().reset_index()
    fig_month = px.line(
        monthly_revenue,
        x="Month",
        y="Revenue",
        markers=True,
        title="Monthly Revenue Trend"
    )
    fig_month.update_layout(
        template="plotly_dark",
        paper_bgcolor="#08111f",
        plot_bgcolor="#08111f",
        font=dict(color="white"),
        title_x=0.02
    )
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown('<div class="section-title">Revenue by Product</div>', unsafe_allow_html=True)
    product_revenue = df.groupby("Product")["Revenue"].sum().reset_index().sort_values(by="Revenue", ascending=False)
    fig_product = px.bar(
        product_revenue,
        x="Product",
        y="Revenue",
        title="Revenue by Product",
        text_auto=".2s"
    )
    fig_product.update_layout(
        template="plotly_dark",
        paper_bgcolor="#08111f",
        plot_bgcolor="#08111f",
        font=dict(color="white"),
        title_x=0.02
    )
    st.plotly_chart(fig_product, use_container_width=True)

# --------------------------------------------------
# SALES ANALYTICS
# --------------------------------------------------
elif menu == "Sales Analytics":
    st.markdown('<div class="section-title">Sales Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle-text">Analyze revenue and quantity using filters for product and category.</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="panel-title">Filters</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        with f1:
            selected_product = st.selectbox("Filter by Product", ["All"] + sorted(df["Product"].unique().tolist()))
        with f2:
            selected_category = st.selectbox("Filter by Category", ["All"] + sorted(df["Category"].unique().tolist()))

    filtered_df = df.copy()
    if selected_product != "All":
        filtered_df = filtered_df[filtered_df["Product"] == selected_product]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["Category"] == selected_category]

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Total Revenue", f"₹ {filtered_df['Revenue'].sum():,.0f}")
    with m2:
        st.metric("Total Quantity Sold", f"{filtered_df['Quantity_Sold'].sum():,}")

    st.markdown('<div class="section-title">Product-wise Revenue</div>', unsafe_allow_html=True)
    product_rev = filtered_df.groupby("Product")["Revenue"].sum().reset_index().sort_values(by="Revenue", ascending=False)
    fig1 = px.bar(product_rev, x="Product", y="Revenue", title="Revenue by Product", text_auto=".2s")
    fig1.update_layout(template="plotly_dark", paper_bgcolor="#08111f", plot_bgcolor="#08111f", font=dict(color="white"), title_x=0.02)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<div class="section-title">Monthly Revenue Trend</div>', unsafe_allow_html=True)
    month_rev = filtered_df.groupby("Month")["Revenue"].sum().reset_index()
    fig2 = px.line(month_rev, x="Month", y="Revenue", markers=True, title="Monthly Revenue")
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#08111f", plot_bgcolor="#08111f", font=dict(color="white"), title_x=0.02)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Quantity Sold by Product</div>', unsafe_allow_html=True)
    qty_prod = filtered_df.groupby("Product")["Quantity_Sold"].sum().reset_index().sort_values(by="Quantity_Sold", ascending=False)
    fig3 = px.bar(qty_prod, x="Product", y="Quantity_Sold", title="Quantity Sold by Product", text_auto=True)
    fig3.update_layout(template="plotly_dark", paper_bgcolor="#08111f", plot_bgcolor="#08111f", font=dict(color="white"), title_x=0.02)
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# SALES PREDICTION
# --------------------------------------------------
elif menu == "Sales Prediction":
    st.markdown('<div class="section-title">Sales Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle-text">Predict future revenue using Linear Regression.</div>', unsafe_allow_html=True)

    X = df[["Year", "Month", "Quantity_Sold", "Unit_Price"]]
    y = df["Revenue"]

    model = LinearRegression()
    model.fit(X, y)

    pred_train = model.predict(X)
    r2 = r2_score(y, pred_train)
    mae = mean_absolute_error(y, pred_train)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown('<div class="panel"><div class="panel-title">Input Details</div><div class="small-note">Enter future sales-related values to estimate expected revenue.</div></div>', unsafe_allow_html=True)
        year = st.number_input("Enter Year", 2023, 2035, 2025)
        month = st.slider("Select Month", 1, 12, 1)
        quantity = st.number_input("Enter Quantity Sold", 1, 5000, 100)
        price = st.number_input("Enter Unit Price", 100, 200000, 50000)
        predict_sales_btn = st.button("Predict Sales")

    with right:
        st.markdown('<div class="panel"><div class="panel-title">Model Performance</div><div class="small-note">Evaluation metrics of the Linear Regression model used for sales forecasting.</div></div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.metric("R² Score", f"{r2:.3f}")
        with b:
            st.metric("MAE", f"{mae:,.2f}")

        st.markdown('<div class="result-info">This module estimates revenue using Year, Month, Quantity Sold, and Unit Price.</div>', unsafe_allow_html=True)

    if predict_sales_btn:
        input_data = pd.DataFrame({
            "Year": [year],
            "Month": [month],
            "Quantity_Sold": [quantity],
            "Unit_Price": [price]
        })
        prediction = model.predict(input_data)[0]
        st.markdown(f'<div class="result-good">Predicted Revenue: ₹ {prediction:,.2f}</div>', unsafe_allow_html=True)

# --------------------------------------------------
# DEMAND PREDICTION
# --------------------------------------------------
elif menu == "Demand Prediction":
    st.markdown('<div class="section-title">Demand Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle-text">Predict monthly product demand using Random Forest.</div>', unsafe_allow_html=True)

    temp_df = df.copy()
    le_product = LabelEncoder()
    le_category = LabelEncoder()

    temp_df["Product_enc"] = le_product.fit_transform(temp_df["Product"])
    temp_df["Category_enc"] = le_category.fit_transform(temp_df["Category"])

    X = temp_df[["Product_enc", "Category_enc", "Stock_Available", "Reorder_Level", "Quantity_Sold"]]
    y = temp_df["Monthly_Demand"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    pred_train = model.predict(X)
    r2 = r2_score(y, pred_train)
    mae = mean_absolute_error(y, pred_train)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown('<div class="panel"><div class="panel-title">Input Details</div><div class="small-note">Choose product details to estimate future monthly demand.</div></div>', unsafe_allow_html=True)
        product = st.selectbox("Select Product", temp_df["Product"].unique())
        category = st.selectbox("Select Category", temp_df["Category"].unique())
        stock = st.number_input("Stock Available", 0, 5000, 120)
        reorder = st.number_input("Reorder Level", 0, 1000, 30)
        quantity = st.number_input("Quantity Sold", 1, 5000, 100)
        predict_demand_btn = st.button("Predict Demand")

    with right:
        st.markdown('<div class="panel"><div class="panel-title">Model Performance</div><div class="small-note">Evaluation metrics of the Random Forest model used for demand forecasting.</div></div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.metric("R² Score", f"{r2:.3f}")
        with b:
            st.metric("MAE", f"{mae:,.2f}")

        st.markdown('<div class="result-info">The model predicts future demand using product, category, stock level, reorder level, and quantity sold.</div>', unsafe_allow_html=True)

    if predict_demand_btn:
        input_data = pd.DataFrame({
            "Product_enc": [le_product.transform([product])[0]],
            "Category_enc": [le_category.transform([category])[0]],
            "Stock_Available": [stock],
            "Reorder_Level": [reorder],
            "Quantity_Sold": [quantity]
        })

        prediction = model.predict(input_data)[0]
        st.markdown(f'<div class="result-good">Predicted Monthly Demand: {prediction:.2f}</div>', unsafe_allow_html=True)

        if prediction > stock:
            shortage = int(np.ceil(prediction - stock))
            st.markdown(f'<div class="result-warn">⚠ Stock Risk! Recommended Reorder Quantity: {shortage}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-good">✅ Stock Level is Sufficient Based on Predicted Demand.</div>', unsafe_allow_html=True)

# --------------------------------------------------
# SMART REORDER INSIGHTS
# --------------------------------------------------
elif menu == "Smart Reorder Insights":
    st.markdown('<div class="section-title">Smart Reorder Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle-text">Automatically identify products that may require stock replenishment.</div>', unsafe_allow_html=True)

    temp_df = df.copy()
    le_product = LabelEncoder()
    le_category = LabelEncoder()

    temp_df["Product_enc"] = le_product.fit_transform(temp_df["Product"])
    temp_df["Category_enc"] = le_category.fit_transform(temp_df["Category"])

    X = temp_df[["Product_enc", "Category_enc", "Stock_Available", "Reorder_Level", "Quantity_Sold"]]
    y = temp_df["Monthly_Demand"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    temp_df["Predicted_Demand"] = model.predict(X)
    temp_df["Shortage"] = np.ceil(temp_df["Predicted_Demand"] - temp_df["Stock_Available"])
    temp_df["Recommendation"] = temp_df["Shortage"].apply(lambda x: "Reorder Needed" if x > 0 else "Stock Sufficient")

    result_df = temp_df[["Product", "Category", "Stock_Available", "Predicted_Demand", "Shortage", "Recommendation"]].copy()
    result_df["Predicted_Demand"] = result_df["Predicted_Demand"].round(2)
    result_df["Shortage"] = result_df["Shortage"].apply(lambda x: int(x) if x > 0 else 0)

    reorder_df = result_df[result_df["Recommendation"] == "Reorder Needed"].sort_values(by="Shortage", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Products Needing Reorder", len(reorder_df))
    with c2:
        max_shortage = reorder_df["Shortage"].max() if not reorder_df.empty else 0
        st.metric("Highest Shortage Quantity", int(max_shortage))

    if not reorder_df.empty:
        st.markdown('<div class="panel-title">Recommended Reorder Table</div>', unsafe_allow_html=True)
        st.dataframe(reorder_df, use_container_width=True)

        fig = px.bar(
            reorder_df,
            x="Product",
            y="Shortage",
            color="Category",
            title="Recommended Reorder Quantity by Product",
            text_auto=True
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#08111f", plot_bgcolor="#08111f", font=dict(color="white"), title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="result-good">All products currently have sufficient stock.</div>', unsafe_allow_html=True)

# --------------------------------------------------
# BUSINESS INSIGHTS
# --------------------------------------------------
elif menu == "Business Insights":
    st.markdown('<div class="section-title">Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle-text">Quick decision-making insights based on sales and product movement.</div>', unsafe_allow_html=True)

    top_products = df.groupby("Product")["Quantity_Sold"].sum().reset_index().sort_values(by="Quantity_Sold", ascending=False)
    slow_products = df.groupby("Product")["Quantity_Sold"].sum().reset_index().sort_values(by="Quantity_Sold", ascending=True)
    revenue_products = df.groupby("Product")["Revenue"].sum().reset_index().sort_values(by="Revenue", ascending=False)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="card"><div class="card-label">Top Selling Product</div><div class="card-value">{top_products.iloc[0]["Product"]}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="card-label">Slow Moving Product</div><div class="card-value">{slow_products.iloc[0]["Product"]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="card-label">Highest Revenue Product</div><div class="card-value">{revenue_products.iloc[0]["Product"]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top 5 Best Selling Products</div>', unsafe_allow_html=True)
    st.dataframe(top_products.head(5), use_container_width=True)
    fig_top = px.bar(top_products.head(5), x="Product", y="Quantity_Sold", title="Top 5 Best Sellers", text_auto=True)
    fig_top.update_layout(template="plotly_dark", paper_bgcolor="#08111f", plot_bgcolor="#08111f", font=dict(color="white"), title_x=0.02)
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown('<div class="section-title">Bottom 5 Slow Moving Products</div>', unsafe_allow_html=True)
    st.dataframe(slow_products.head(5), use_container_width=True)
    fig_slow = px.bar(slow_products.head(5), x="Product", y="Quantity_Sold", title="Bottom 5 Slow Moving Products", text_auto=True)
    fig_slow.update_layout(template="plotly_dark", paper_bgcolor="#08111f", plot_bgcolor="#08111f", font=dict(color="white"), title_x=0.02)
    st.plotly_chart(fig_slow, use_container_width=True)

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------
elif menu == "About Project":
    st.markdown('<div class="section-title">About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Project Summary</div>
        <div class="small-note">
            The AI-Based Smart ERP Analytics System is designed to improve business decision-making
            using ERP-style sales and inventory data. It combines data analytics with machine learning
            to monitor performance, forecast revenue, predict future demand, and generate smart stock insights.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Modules Included</div>
        <div class="small-note">
            • Dashboard Overview<br>
            • Sales Analytics<br>
            • Sales Prediction<br>
            • Demand Prediction<br>
            • Smart Reorder Insights<br>
            • Business Insights
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Machine Learning Models Used</div>
        <div class="small-note">
            • Linear Regression for Sales Prediction<br>
            • Random Forest Regressor for Demand Prediction
        </div>
    </div>
    """, unsafe_allow_html=True)