import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


filepath = ("polynomial_regression.csv")
df = pd.read_csv(filepath)

np.random.seed(25)

x = df[['x']]
y = df['y']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=29)

no_of_samples = 30
sample_size = 20
degree_of_polynomial = np.arange(1, 11)

train_data_error = []
test_data_error = []

train_mse_data = []
test_mse_data = []


for sample in range(no_of_samples):
    indices = np.random.choice(len(x_train), sample_size, replace=False)
    x_sample = x_train.iloc[indices]
    y_sample = y_train.iloc[indices]

    train_sample_error = []
    test_sample_error = []
    train_mse_sample = []
    test_mse_sample = []

    for degree in range(1, 11):
        polynomial = PolynomialFeatures(degree=degree)
        x_train_poly = polynomial.fit_transform(x_sample)
        x_test_poly = polynomial.transform(x_test)

        model = LinearRegression()
        model.fit(x_train_poly, y_sample)

        y_train_predict = model.predict(x_train_poly)
        y_test_predict = model.predict(x_test_poly)

        train_r2 = r2_score(y_sample, y_train_predict)
        test_r2 = r2_score(y_test, y_test_predict)

        train_mse = mean_squared_error(y_sample, y_train_predict)
        test_mse = mean_squared_error(y_test, y_test_predict)

        train_sample_error.append(train_r2)
        test_sample_error.append(test_r2)

        train_mse_sample.append(train_mse)
        test_mse_sample.append(test_mse)

    train_data_error.append(train_sample_error)
    test_data_error.append(test_sample_error)

    train_mse_data.append(train_mse_sample)
    test_mse_data.append(test_mse_sample)

# Convert to numpy arrays
train_data_error = np.array(train_data_error)
test_data_error = np.array(test_data_error)

average_train_r2 = np.mean(train_data_error, axis=0)
average_test_r2 = np.mean(test_data_error, axis=0)

print("\nAverage R² scores over all samples:")
for degree, train_r2, test_r2 in zip(degree_of_polynomial, average_train_r2, average_test_r2):
    print(f"Degree {degree}: Average Train R² = {train_r2:.4f}, Average Test R² = {test_r2:.4f}")

# DataFrames for violin plots
train_err_df = pd.DataFrame(train_data_error, columns=degree_of_polynomial)
test_err_df = pd.DataFrame(test_data_error, columns=degree_of_polynomial)
difference = train_err_df - test_err_df

# Violin plot for test R²
plt.figure(figsize=(12,6))
sns.violinplot(data=test_err_df)
plt.xlabel("Polynomial Degree")
plt.ylabel("Test data R²")
plt.title("Test R² Fluctuations Across Degrees")
plt.show()

# Violin plot for train-test R² difference
plt.figure(figsize=(12,6))
sns.violinplot(data=difference)
plt.xlabel("Polynomial Degree")
plt.ylabel("Train - Test R²")
plt.title("Train-Test R² Difference Across Degrees")
plt.show()

# MSE Violin Plots in Log Scale
train_mse_df = pd.DataFrame(train_mse_data, columns=degree_of_polynomial)
test_mse_df = pd.DataFrame(test_mse_data, columns=degree_of_polynomial)

# Convert to log10 scale (add epsilon to avoid log(0))
epsilon = 1e-8
log_train_mse_df = np.log10(train_mse_df + epsilon)
log_test_mse_df = np.log10(test_mse_df + epsilon)

# Log-scaled Test MSE violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=log_test_mse_df)
plt.xlabel("Polynomial Degree")
plt.ylabel("Log10(Test MSE)")
plt.title("Log-Scaled Test MSE vs. Polynomial Degree")
plt.show()

mse_difference = train_mse_df - test_mse_df
log_mse_difference = np.log10(np.abs(mse_difference) + epsilon)

# Log-scaled MSE Difference violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=log_mse_difference)
plt.xlabel("Polynomial Degree")
plt.ylabel("Log10(|Train MSE - Test MSE|)")
plt.title("Log-Scaled MSE Difference vs. Polynomial Degree")
plt.show()

# Cross-validation to choose best degree on sample
new_sample_index = np.random.choice(len(x_train), sample_size, replace=False)
new_sample_x = x_train.iloc[new_sample_index]
new_sample_y = y_train.iloc[new_sample_index] 

kf = KFold(n_splits=5, shuffle=True, random_state=67)
cv_scores = []

print("\nMean CV R² for each degree:\n")
for degree in range(1, 11):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(new_sample_x)

    model = LinearRegression()
    scores = cross_val_score(model, x_poly, new_sample_y, cv=kf, scoring='r2')
    cv_scores.append(scores.mean())

    print(f"Degree {degree}: {scores.mean():.4f}")

best_degree = np.argmax(cv_scores) + 1
print("Best degree from 5-fold CV on sample of 20 points:", best_degree)

# Train final model using the best degree
polynomial_best = PolynomialFeatures(degree=best_degree)
x_sample_poly = polynomial_best.fit_transform(new_sample_x)
final_model = LinearRegression()
final_model.fit(x_sample_poly, new_sample_y)

# Test performance of final model
x_test_final = polynomial_best.transform(x_test)
y_test_final_pred = final_model.predict(x_test_final)
final_test_r2 = r2_score(y_test, y_test_final_pred)
print("R² on test data using the best degree", best_degree, ":", final_test_r2)

kf_full = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_full = []

print("\nFinding best degree using full training data:\n")
for degree in range(1, 11):
    poly = PolynomialFeatures(degree=degree)
    x_poly_full = poly.fit_transform(x_train)

    model = LinearRegression()
    scores = cross_val_score(model, x_poly_full, y_train, cv=kf_full, scoring='r2')
    cv_scores_full.append(scores.mean())

    print(f"Degree {degree}: {scores.mean():.4f}")

best_degree_full = np.argmax(cv_scores_full) 
print("Best degree from 5-fold CV on full train set:", best_degree_full)

# Train final model on full train set with best degree
polynomial_full_best = PolynomialFeatures(degree=best_degree_full)
x_train_poly_full_best = polynomial_full_best.fit_transform(x_train)
final_model_full = LinearRegression()
final_model_full.fit(x_train_poly_full_best, y_train)

# Test performance
x_test_poly_full_best = polynomial_full_best.transform(x_test)
y_test_pred_full_best = final_model_full.predict(x_test_poly_full_best)
final_test_r2_full = r2_score(y_test, y_test_pred_full_best)
print("R² on test data using the best degree from full train set:", final_test_r2_full)

# Ridge and Lasso Regularization
alphas = [1, 0.1, 0.01, 0.001]
kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)

ridge_cv_scores, lasso_cv_scores = [], []

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(x_train_poly_full_best, y_train)
    ridge_coefs = ridge.coef_
    print(f"Ridge Coefficients (alpha={alpha}):", ridge_coefs)
    ridge_scores = cross_val_score(Ridge(alpha=alpha), x_train_poly_full_best, y_train, cv=kf_10, scoring='r2')
    ridge_cv_scores.append(ridge_scores.mean())

    # Lasso
    lasso = Lasso(alpha=alpha)
    lasso.fit(x_train_poly_full_best, y_train)
    lasso_coefs = lasso.coef_
    print(f"Lasso Coefficients (alpha={alpha}):", lasso_coefs)
    lasso_scores = cross_val_score(Lasso(alpha=alpha), x_train_poly_full_best, y_train, cv=kf_10, scoring='r2')
    lasso_cv_scores.append(lasso_scores.mean())

# Find best alpha
best_alpha_ridge = alphas[np.argmax(ridge_cv_scores)]
best_alpha_lasso = alphas[np.argmax(lasso_cv_scores)]

print("\nBest alpha for Ridge:", best_alpha_ridge)
print("Best alpha for Lasso:", best_alpha_lasso)

# Train final Ridge model with best alpha
ridge_best = Ridge(alpha=best_alpha_ridge)
ridge_best.fit(x_train_poly_full_best, y_train)
ridge_test_r2 = r2_score(y_test, ridge_best.predict(x_test_poly_full_best))
print("Ridge Test R²:", ridge_test_r2)

# Train final Lasso model with best alpha
lasso_best = Lasso(alpha=best_alpha_lasso)
lasso_best.fit(x_train_poly_full_best, y_train)
lasso_test_r2 = r2_score(y_test, lasso_best.predict(x_test_poly_full_best))
print("Lasso Test R²:", lasso_test_r2)


