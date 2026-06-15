import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the Dataset
print("📥 Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Let's look at the first few rows
print("\n--- Dataset Preview ---")
print(df.head())

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# 3. Split the Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train the Model
print("\n Training the predictivemo")# Import Libraries
import ssl
import pandas as pd

# Fix SSL Certificate Issue
ssl._create_default_https_context = ssl._create_unverified_context

# Import Dataset and ML Tools
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
print("Loading California Housing Dataset...\n")

housing = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add Target Column
df["MedHouseVal"] = housing.target

# Dataset Preview
print("----- Dataset Preview -----")
print(df.head())

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# 3. Split the Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Feature Scaling
print("\nScaling Features...\n")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train the Model
print("Training the Predictive Model...\n")

model = LinearRegression()

model.fit(X_train_scaled, y_train)

print("Model Trained Successfully!\n")

# 6. Make Predictions
predictions = model.predict(X_test_scaled)

print("----- Predictions -----")
print(predictions[:5])

# 7. Evaluate the Model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n----- Model Evaluation -----")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)