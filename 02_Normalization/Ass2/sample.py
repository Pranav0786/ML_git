import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
from sklearn.preprocessing import quantile_transform

# Generate Data
gaussian_mean = 5
gaussian_sd = 2
totalsize = 10000

np.random.seed(69)

# Gaussian distribution
B = np.random.normal(gaussian_mean, gaussian_sd, size=totalsize)

# Power Law distribution
a = 0.3
I = powerlaw.rvs(a, size=totalsize)

# Geometric Distribution
p = 0.005
H = np.random.geometric(p, size=totalsize)

# Function for plotting histogram comparison
def plot_histogram(original, normalized, title):
    plt.hist(original, bins=100, alpha=0.5, label='Original', density=True)
    plt.hist(normalized, bins=100, alpha=0.5, label='Normalized', density=True)
    plt.title(title)
    plt.legend()
    plt.show()

# Function for plotting single boxplot for all normalized versions
def plot_all_boxplots(normalized_values, labels, title):
    plt.boxplot(normalized_values, labels=labels)
    plt.title(title)
    plt.show()

# 1. Max Normalization
def max_normalization(B, I, H):
    return B / np.max(B), I / np.max(I), H / np.max(H)

Gaussian_max, powerlaw_max, geometric_max = max_normalization(B, I, H)

# Histogram comparison
plot_histogram(B, Gaussian_max, "Max Normalization - Gaussian")
plot_histogram(I, powerlaw_max, "Max Normalization - Powerlaw")
plot_histogram(H, geometric_max, "Max Normalization - Geometric")

# 2. Sum Normalization
def sum_normalization(B, I, H):
    return B / np.sum(B), I / np.sum(I), H / np.sum(H)

Gaussian_sum, powerlaw_sum, geometric_sum = sum_normalization(B, I, H)

plot_histogram(B, Gaussian_sum, "Sum Normalization - Gaussian")
plot_histogram(I, powerlaw_sum, "Sum Normalization - Powerlaw")
plot_histogram(H, geometric_sum, "Sum Normalization - Geometric")

# 3. Z-Score Normalization
def zscore_normalization(B, I, H):
    return (B - B.mean()) / B.std(), (I - I.mean()) / I.std(), (H - H.mean()) / H.std()

Gaussian_zscore, powerlaw_zscore, geometric_zscore = zscore_normalization(B, I, H)

plot_histogram(B, Gaussian_zscore, "Z-Score Normalization - Gaussian")
plot_histogram(I, powerlaw_zscore, "Z-Score Normalization - Powerlaw")
plot_histogram(H, geometric_zscore, "Z-Score Normalization - Geometric")

# 4. Percentile Normalization
def percentile_normalization(B, I, H):
    def percentile(arr):
        return (np.arange(1, len(arr) + 1) / totalsize) * 100
    return percentile(B), percentile(I), percentile(H)

gaussian_percentile, powerlaw_percentile, geometric_percentile = percentile_normalization(B, I, H)

plot_histogram(B, gaussian_percentile, "Percentile Normalization - Gaussian")
plot_histogram(I, powerlaw_percentile, "Percentile Normalization - Powerlaw")
plot_histogram(H, geometric_percentile, "Percentile Normalization - Geometric")

# 5. Median Normalization
def same_median(B, I, H):
    median_target = np.median([np.median(B), np.median(I), np.median(H)])
    return (B * (median_target / np.median(B))),(I*(median_target / np.median(I))),(H * (median_target / np.median(H)))

new_B, new_I, new_H = same_median(B, I, H)

plot_histogram(B, new_B, "Same Median Normalization - Gaussian")
plot_histogram(I, new_I, "Same Median Normalization - Powerlaw")
plot_histogram(H, new_H, "Same Median Normalization - Geometric")

# 6. Quantile Normalization using off-the-shelf function
df = pd.DataFrame({"Gaussian": B, "Powerlaw": I, "Geometric": H})

df_quantile_normalized = pd.DataFrame(
    quantile_transform(df, output_distribution='normal', copy=True),
    columns=df.columns
)

plot_histogram(B, df_quantile_normalized["Gaussian"], "Quantile Normalization - Gaussian")
plot_histogram(I, df_quantile_normalized["Powerlaw"], "Quantile Normalization - Powerlaw")
plot_histogram(H, df_quantile_normalized["Geometric"], "Quantile Normalization - Geometric")

# Compare all normalized variables in a single box plot
plot_all_boxplots(
    [Gaussian_max, powerlaw_max, geometric_max],
    ["Gaussian", "Powerlaw", "Geometric"],
    "Boxplot - Max Normalization"
)

plot_all_boxplots(
    [Gaussian_sum, powerlaw_sum, geometric_sum],
    ["Gaussian", "Powerlaw", "Geometric"],
    "Boxplot - Sum Normalization"
)

plot_all_boxplots(
    [Gaussian_zscore, powerlaw_zscore, geometric_zscore],
    ["Gaussian", "Powerlaw", "Geometric"],
    "Boxplot - Z-Score Normalization"
)

plot_all_boxplots(
    [gaussian_percentile, powerlaw_percentile, geometric_percentile],
    ["Gaussian", "Powerlaw", "Geometric"],
    "Boxplot - Percentile Normalization"
)

plot_all_boxplots(
    [new_B, new_I, new_H],
    ["Gaussian", "Powerlaw", "Geometric"],
    "Boxplot - Same Median Normalization"
)

plot_all_boxplots(
    [df_quantile_normalized["Gaussian"], df_quantile_normalized["Powerlaw"], df_quantile_normalized["Geometric"]],
    ["Gaussian", "Powerlaw", "Geometric"],
    "Boxplot - Quantile Normalization"
)
