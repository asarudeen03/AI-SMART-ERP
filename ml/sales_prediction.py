# sales_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv("data/Smart_ERP_Final_Dataset.csv")

# Step 2: Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Step 3: Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Step 4: Select Features & Target
X = df[['Year', 'Month', 'Quantity_Sold', 'Unit_Price']]

y = df['Revenue']

# Step 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Step 9: Example Prediction
example = pd.DataFrame({
    'Year': [2025],
    'Month': [1],
    'Quantity_Sold': [150],
    'Unit_Price': [50000]
})


  # Year, Month, Quantity
predicted_sales = model.predict(example)

print("\nPredicted Sales for Jan 2025 with 150 quantity:", predicted_sales[0])
