import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
from scipy.stats import rankdata
from sklearn.preprocessing import quantile_transform

gaussian_mean = 5
gaussian_sd = 2
totalsize = 10000

np.random.seed(45)

# Gaussian distribution
B = np.random.normal(gaussian_mean, gaussian_sd, size=totalsize)

# Power Law distribution
a = 0.3
I = powerlaw.rvs(a, size=totalsize)

# Geometric Distribution
p = 0.005
H = np.random.geometric(p, size=totalsize)


def plot_histogram(data_list, labels, title):
    plt.hist(data_list, bins=100, label=labels, alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()


plt.boxplot([B, I, H], labels=["Gaussian", "Powerlaw", "Geometric"])
plt.title("Box-plot distribution")
plt.show()

plot_histogram([B], ["Gaussian"], "Histogram distribution of Gaussian variable")
plot_histogram([I], ["Powerlaw"], "Histogram distribution of Powerlaw variable")
plot_histogram([H], ["Geometric"], "Histogram distribution of Geometric variable")

# Max Normalization
def max_normalization(B, I, H):
    return B / np.max(B), I / np.max(I), H / np.max(H)

Gaussian_max, powerlaw_max, geometric_max = max_normalization(B, I, H)

plt.boxplot([Gaussian_max, powerlaw_max, geometric_max], labels=["Gaussian", "Powerlaw", "Geometric"])
plt.title("Box-plot distribution max_normalization")
plt.show()

plot_histogram([B, Gaussian_max], ["Original-Gaussian", "Normalized-Gaussian"], "Histogram distribution of Gaussian variable")
plot_histogram([I, powerlaw_max], ["Original-Powerlaw", "Normalized-Powerlaw"], "Histogram distribution of Powerlaw variable")
plot_histogram([H, geometric_max], ["Original-Geometric", "Normalized-Geometric"], "Histogram distribution of Geometric variable")

# 2. Sum Normalization
def sum_normalization(B, I, H):
    return B / np.sum(B), I / np.sum(I), H / np.sum(H)

Gaussian_sum, powerlaw_sum, geometric_sum = sum_normalization(B, I, H)

plt.boxplot([Gaussian_sum, powerlaw_sum, geometric_sum], labels=["Gaussian", "Powerlaw", "Geometric"])
plt.title("Box-plot distribution sum_normalization")
plt.show()

plot_histogram([B, Gaussian_sum], ["Original-Gaussian", "Normalized-Gaussian"], "Histogram distribution of Gaussian variable")
plot_histogram([I, powerlaw_sum], ["Original-Powerlaw", "Normalized-Powerlaw"], "Histogram distribution of Powerlaw variable")
plot_histogram([H, geometric_sum], ["Original-Geometric", "Normalized-Geometric"], "Histogram distribution of Geometric variable")

# Z-score Normalization
def zscore_normalization(B, I, H):
    B_zscore = (B - B.mean()) / B.std()
    I_zscore = (I - I.mean()) / I.std()
    H_zscore = (H - H.mean()) / H.std()
    return B_zscore,I_zscore ,H_zscore 

Gaussian_zscore, powerlaw_zscore, geometric_zscore = zscore_normalization(B, I, H)

plt.boxplot([Gaussian_zscore, powerlaw_zscore, geometric_zscore], labels=["Gaussian", "Powerlaw", "Geometric"])
plt.title("Box-plot of z-score Normalized Variables")
plt.show()

plot_histogram([B, Gaussian_zscore], ["Original-Gaussian", "Normalized-Gaussian"], "Histogram distribution of Gaussian variable")
plot_histogram([I, powerlaw_zscore], ["Original-Powerlaw", "Normalized-Powerlaw"], "Histogram distribution of Powerlaw variable")
plot_histogram([H, geometric_zscore], ["Original-Geometric", "Normalized-Geometric"], "Histogram distribution of Geometric variable")

# Percentile Normalization
def percentile(array):
    curr_value = rankdata(array, method='average')
    percentile_value = (curr_value / totalsize) * 100
    return percentile_value

def percentile_normalization(B, I, H):
    return percentile(B), percentile(I), percentile(H)

gaussian_percentile, powerlaw_percentile, geometric_percentile = percentile_normalization(B, I, H)

plt.boxplot([gaussian_percentile, powerlaw_percentile, geometric_percentile], labels=["Gaussian", "Powerlaw", "Geometric"])
plt.title("Box-plot of percentile normalized variables")
plt.show()

plot_histogram([B, gaussian_percentile], ["Original-Gaussian", "Normalized-Gaussian"], "Histogram distribution of Gaussian variable")
plot_histogram([I, powerlaw_percentile], ["Original-Powerlaw", "Normalized-Powerlaw"], "Histogram distribution of Powerlaw variable")
plot_histogram([H, geometric_percentile], ["Original-Geometric", "Normalized-Geometric"], "Histogram distribution of Geometric variable")

# 5. Same Median Normalization
def same_median(B, I, H):
    gaussian_median = np.median(B)
    powerlaw_median = np.median(I)
    geometric_median = np.median(H)
    mean_of_median = (gaussian_median + powerlaw_median + geometric_median) / 3

    return (mean_of_median / gaussian_median) * B, (mean_of_median / powerlaw_median) * I, (mean_of_median / geometric_median) * H

new_B, new_I, new_H = same_median(B, I, H)

plt.boxplot([new_B, new_I, new_H], labels=["Gaussian", "Powerlaw", "Geometric"])
plt.title("Box-plot after making median same")
plt.show()

plot_histogram([B, new_B], ["Original-Gaussian", "Normalized-Gaussian"], "Histogram distribution of Gaussian variable")
plot_histogram([I, new_I], ["Original-Powerlaw", "Normalized-Powerlaw"], "Histogram distribution of Powerlaw variable")
plot_histogram([H, new_H], ["Original-Geometric", "Normalized-Geometric"], "Histogram distribution of Geometric variable")

# 6. Quantile Normalization
dataframe = pd.DataFrame(
    {   
        "Gaussian": B, 
        "Powerlaw": I, 
        "Geometric": H
    }
)

dataframe_quantile_normalized = pd.DataFrame(
    quantile_transform(dataframe, output_distribution='normal', copy=True),
    columns=dataframe.columns
)

plt.boxplot([dataframe_quantile_normalized["Gaussian"], dataframe_quantile_normalized["Powerlaw"], dataframe_quantile_normalized["Geometric"]], 
            labels=dataframe.columns)
plt.title("Box-plot of Quantile Normalized Variables (Using Off-the-Shelf Library)")
plt.show()

plot_histogram([B, dataframe_quantile_normalized["Gaussian"]], ["Original", "Quantile-Gaussian"], "Histogram of Gaussian Quantile Normalization")
plot_histogram([I, dataframe_quantile_normalized["Powerlaw"]], ["Original", "Quantile-Powerlaw"], "Histogram of Powerlaw Quantile Normalization")
plot_histogram([H, dataframe_quantile_normalized["Geometric"]], ["Original", "Quantile-Geometric"], "Histogram of Geometric Quantile Normalization")
