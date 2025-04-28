import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('linear_regression_3.csv')

# Split Data (80-20)
X = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']]
y = df['y']

print(f"\n {X}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8675309)

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Train initial model
model_train = sm.OLS(y_train, X_train_sm).fit()
print("\nTrain R2 (initial):", model_train.rsquared)
print("Train Adjusted R2 (initial):", model_train.rsquared_adj)

# Remove outliers using Cook's distance
influence = OLSInfluence(model_train)
cooks = influence.cooks_distance[0]
mask = cooks < (4 / len(X_train_sm))
X_train_sm = X_train_sm[mask]
y_train = y_train[mask]
print(f"\nOutliers removed from training: {len(X_train) - len(X_train_sm)}")

model_train = sm.OLS(y_train, X_train_sm).fit()
print("\nTrain R2 (after outlier removal):", model_train.rsquared)
print("Train Adjusted R2 (after outlier removal):", model_train.rsquared_adj)

# Calculate and remove high VIF features 
def compute_vif(X):
    vif_dict = {col: variance_inflation_factor(X.values, i) 
                for i, col in enumerate(X.columns) if col != 'const'}
    return pd.DataFrame({'feature': list(vif_dict.keys()), 'VIF': list(vif_dict.values())})

remaining_columns = list(X_train_sm)
removed_features = []

while True:
    vif_df = compute_vif(X_train_sm[remaining_columns])
    max_vif = vif_df['VIF'].max()
    print("Current VIF values:\n", vif_df, "\n")
    if max_vif <= 5:
        break
    feature = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
    remaining_columns.remove(feature)
    removed_features.append(feature)

model_vif = sm.OLS(y_train, X_train_sm[remaining_columns]).fit()
print(f"\nFeatures removed due to high VIF: {len(removed_features)}")
print("Features removed: ", removed_features)
print("Train R2 (after VIF removal):", model_vif.rsquared)
print("Train Adjusted R2 (after VIF removal):", model_vif.rsquared_adj)

# Test set evaluation
X_test_full = X_test_sm[remaining_columns]
y_pred = model_vif.predict(X_test_full)
residuals = y_test - y_pred

r2_test = model_vif.rsquared 
n, p = len(y_test), model_vif.df_model
adj_r2_test = model_vif.rsquared_adj

print("\nTest R2:", r2_test)
print("Test Adjusted R2:", adj_r2_test)

plt.figure()
plt.hist(residuals, bins=30, edgecolor='k')
plt.title("Test Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# Remove test set outliers
Q1, Q3 = np.percentile(residuals, [25, 75])
IQR = Q3 - Q1
mask_test = (residuals >= Q1 - 1.5*IQR) & (residuals <= Q3 + 1.5*IQR)
X_test_clean = X_test_full[mask_test]
y_test_clean = y_test[mask_test]

y_pred_clean = model_vif.predict(X_test_clean)
residuals_clean = y_test_clean - y_pred_clean

r2_clean = 1 - (np.sum(residuals_clean**2) / np.sum((y_test_clean - np.mean(y_test_clean))**2))
n_clean = len(y_test_clean)
adj_r2_clean = 1 - ((1 - r2_clean) * (n_clean - 1) / (n_clean - p - 1))

print(f"\nOutliers removed from test set: {len(X_test_full) - len(X_test_clean)}")
print("Test R2 (cleaned):", r2_clean)
print("Test Adjusted R2 (cleaned):", adj_r2_clean)