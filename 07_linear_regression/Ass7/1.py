import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

file_path = "linear_regression_3.csv"
df = pd.read_csv(file_path)

X = df.drop(['y'], axis=1)
Y = df['y']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#Remove Outliers using IQR method
def remove_outliers_iqr(X, Y):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    index = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
    
    return X[index], Y[index]


x_train, y_train = remove_outliers_iqr(x_train, y_train)
print("outliers removed using iqr: ", 800-len(x_train))


#Remove features with high VIF (VIF > 4)
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def remove_high_vif(X, vif_threshold):
    while True:
        vif_data = calculate_vif(X)
        print("Current VIF values:\n", vif_data, "\n") 
        
        max_vif = vif_data["VIF"].max()
        if max_vif <= vif_threshold:
            break 

        max_vif_index = vif_data["VIF"].idxmax()
        print(f"max-index: {max_vif_index}\n")
        high_vif_feature = vif_data.loc[max_vif_index, "Feature"]
        print(f"Removing feature: {high_vif_feature}, VIF: {max_vif}\n")
        
        X = X.drop(columns=[high_vif_feature])
    
    return X

X_vif_clean = remove_high_vif(x_train, 10.0)
print(len(X_vif_clean))

X_vif_const = sm.add_constant(X_vif_clean)


# Detect Influence Points using Cook's Distance and DFFITS 
 
model = sm.OLS(y_train, X_vif_const).fit()
print(f"model summary: {model.summary()}")

influence = model.get_influence()
cooks_d, _ = influence.cooks_distance
# print("cooks distances, count: ", cooks_d, len(cooks_d))
threshold_cook = 4 / len(X_vif_clean)

dffits_values, _ = influence.dffits
threshold_dffits = 1.1 * np.sqrt(X_vif_clean.shape[1] / len(X_vif_clean))

# Detect high influence points
high_influence_points = (cooks_d > threshold_cook) | (np.abs(dffits_values) > threshold_dffits)

X_final_train = X_vif_clean[~high_influence_points]
y_final_train = y_train[~high_influence_points]


linear_model = LinearRegression()
linear_model.fit(X_final_train, y_final_train)

print("X_final_train.columns: ", X_final_train.columns)

y_test_pred = linear_model.predict(x_test[X_final_train.columns])  
r2_test = r2_score(y_test, y_test_pred)
print(f"R² Score on original test dataset: {r2_test:.4f}")


# After removing high influence points
y_train_pred = linear_model.predict(X_final_train)
r2_train = r2_score(y_final_train, y_train_pred)
print(f"R² Score on train dataset after removing influence points: {r2_train:.4f}")




# Residuals on test data set
residuals = y_test - y_test_pred

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("residuals")
plt.ylabel("Freq")
plt.title(" residuals on test data")
plt.show()

# IQR method to detect outlier in residuals
Q1_res = residuals.quantile(0.25)
Q3_res = residuals.quantile(0.75)
IQR_res = Q3_res - Q1_res
lower_bound_res = Q1_res - 1.5 * IQR_res
upper_bound_res = Q3_res + 1.5 * IQR_res

# Filter non-outlier test data
non_outlier_index = (residuals >= lower_bound_res) & (residuals <= upper_bound_res)

y_test_clean_no_outliers = y_test[non_outlier_index]
y_test_pred_no_outliers = y_test_pred[non_outlier_index]

# R² on test data without outlier residuals
r2_test_no_outliers = r2_score(y_test_clean_no_outliers, y_test_pred_no_outliers)
print(f"R² Score on test dataset after removing outlier residuals: {r2_test_no_outliers:.4f}")