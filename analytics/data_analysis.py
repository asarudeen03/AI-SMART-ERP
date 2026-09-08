# data_analysis.py

import pandas as pd


def calculate_total_revenue(df):
    return df["Revenue"].sum()


def calculate_total_quantity(df):
    return df["Quantity_Sold"].sum()


def get_revenue_by_product(df):
    return df.groupby("Product")["Revenue"].sum()


def get_monthly_revenue(df):
    return df.groupby("Month")["Revenue"].sum()
