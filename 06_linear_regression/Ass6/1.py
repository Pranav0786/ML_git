import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore  # Import Z-score function

# Load CSV file
df = pd.read_csv("linear_regression_3.csv")  # Replace with actual file path

# Check for missing values
print(df.isnull().sum())

# Ensure X is a DataFrame and y is a Series
y = df.iloc[:, 0]        # First column is 'y' (Series)
X = df.iloc[:, 1:].copy()  # Rest are features (DataFrame)

# ---- Step 1: Remove Outliers using Z-score ----
df_clean = df[(np.abs(zscore(df)) < 3).all(axis=1)]  # Keep only rows with Z < 3

# Extract cleaned X and y
y = df_clean.iloc[:, 0]
X = df_clean.iloc[:, 1:].copy()

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Plot: Actual vs Predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual y values")
plt.ylabel("Predicted y values")
plt.title("Actual vs Predicted Values")
plt.show()
