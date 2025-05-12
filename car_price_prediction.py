import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate  # For structured output
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 📌 Load dataset
file_path = "car_price_prediction_.csv"
df = pd.read_csv(file_path)

# 📌 Display dataset information (Similar to df.info())
print("# Dataset Information\n")
print(df.info())

# 📌 Find and display missing values
print("\n# Find Null Values")
missing_values = df.isnull().sum()
print(missing_values)

# 📌 Display missing values in table format
print("\n🔹 Missing Values in Dataset:")
print(tabulate(missing_values.to_frame(), headers=["Column", "Missing Values"], tablefmt="grid"))

# 📌 Handling missing values
df.dropna(inplace=True)  # Dropping missing values for simplicity

# ✅ **Fix: Correct target column name**
target_column = "Price"  # Use "Price" instead of "price"

# 📌 Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 📌 Scale numerical features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove(target_column)  # Ensure "Price" is excluded

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 📌 Split data into training and testing sets
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Predict on test set
num_samples = 5
sample_cars = X_test.iloc[:num_samples].copy()
predicted_prices = model.predict(sample_cars)

# 📌 Add predictions to table
sample_cars.loc[:, "Predicted Price"] = predicted_prices

# 📌 Reverse scaling for better readability
if numerical_cols:
    sample_cars.loc[:, numerical_cols] = scaler.inverse_transform(sample_cars[numerical_cols])

# 📌 Display results in a structured table format
print("\n🔹 Predicted Car Prices:")
print(tabulate(sample_cars, headers="keys", tablefmt="grid"))
