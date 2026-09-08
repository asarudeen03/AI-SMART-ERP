print("File is running...")
# demand_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load Dataset
df = pd.read_csv("data/Smart_ERP_Final_Dataset.csv")

# Step 2: Encode Categorical Variables
le_product = LabelEncoder()
le_category = LabelEncoder()

df['Product'] = le_product.fit_transform(df['Product'])
df['Category'] = le_category.fit_transform(df['Category'])

# Step 3: Define Features & Target
X = df[['Product', 'Category', 'Stock_Available', 'Reorder_Level', 'Quantity_Sold']]

y = df['Monthly_Demand']

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Demand Prediction Model Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Step 8: Example Prediction
example = pd.DataFrame({
    'Product': [le_product.transform(['Laptop'])[0]],
    'Category': [le_category.transform(['Electronics'])[0]],
    'Stock_Available': [120],
    'Reorder_Level': [30],
    'Quantity_Sold': [100]   # <-- ADD THIS
})


predicted_demand = model.predict(example)

print("\nPredicted Monthly Demand:", predicted_demand[0])

# Step 9: Low Stock Alert Logic
if 120 < 30:
    print("⚠ Low Stock Alert: Reorder Needed!")
else:
    print("Stock Level is Safe.")
