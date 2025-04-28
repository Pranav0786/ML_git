import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
from sklearn.preprocessing import quantile_transform

gaussian_mean = 5
gaussian_sd = 2
totalsize = 10000


#Gaussian distribution
B = np.random.normal(gaussian_mean ,gaussian_sd , size = totalsize)


#Power Law distribution
a = 0.3
I = powerlaw.rvs(a, size = totalsize)


# #Geometric Distribution
p = 0.005
H = np.random.geometric( p, size = totalsize)

plt.boxplot([B,I,H], tick_labels=["Gaussian","powerlaw","geometric"])
plt.title("Box-plot distribution")
plt.show()

plt.hist(B, bins=100, label="Gaussian", edgecolor= "gray", color="pink")
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of gaussian variable")
plt.show()

plt.hist(I, bins=100, label="Powerlaw", edgecolor= "gray")
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of powerlaw variable")
plt.show()

plt.hist( H, bins=100, label="Geometric", edgecolor= "gray", color="green")
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of geometric variable")
plt.show()



# Divide each variable by max
def max_normalization(B, I , H):
    Gaussian_max = B/np.max(B)
    powerlaw_max = I/np.max(I)
    geometric_max = H/np.max(H)
    return Gaussian_max, powerlaw_max, geometric_max

Gaussian_max, powerlaw_max, geometric_max = max_normalization(B, I, H)

plt.boxplot([Gaussian_max, powerlaw_max, geometric_max], tick_labels=["Gaussian","powerlaw","geometric"])
plt.title("Box-plot distribution max_normalization")
plt.show()

plt.hist( [B,Gaussian_max], bins=100, label=["original-Gaussian", "Normalized-Gaussian"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of gaussian variable")
plt.show()

plt.hist( [I,powerlaw_max], bins=100, label=["original-Powerlaw","Normalized-powerlaw"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of powerlaw variable")
plt.show()

plt.hist( [H,geometric_max], bins=100, label=["original-Geometric","Normalized-Geometric"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of geometric variable")
plt.show()


# Divide each variable by sum of its values
def sum_normalization(B , I, H):
    Gaussian_sum = B/np.sum(B)
    powerlaw_sum = I/np.sum(I)
    geometric_sum = H/np.sum(H)
    return Gaussian_sum, powerlaw_sum, geometric_sum

Gaussian_sum, powerlaw_sum, geometric_sum = sum_normalization(B, I, H)

plt.boxplot([Gaussian_sum, powerlaw_sum, geometric_sum], tick_labels=["Gaussian","powerlaw","geometric"])
plt.title("Box-plot distribution sum_normalization")
plt.show()

plt.hist( [B,Gaussian_sum], bins=100, label=["original-Gaussian", "Normalized-Gaussian"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of gaussian variable")
plt.show()

plt.hist( [I,powerlaw_sum], bins=100, label=["original-Powerlaw","Normalized-powerlaw"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of powerlaw variable")
plt.show()

plt.hist( [H,geometric_sum], bins=100, label=["original-Geometric","Normalized-Geometric"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of geometric variable")
plt.show()



# # Convert each variable into z score using respective mean and sd
def zscore_normalization(B, I, H):
    Gaussian_zscore = (B-B.mean())/B.std()
    powerlaw_zscore = (I-I.mean())/I.std()
    geometric_zscore = (H-H.mean())/H.std()
    return Gaussian_zscore, powerlaw_zscore, geometric_zscore

Gaussian_zscore, powerlaw_zscore, geometric_zscore = zscore_normalization(B, I,H)

plt.boxplot([Gaussian_zscore, powerlaw_zscore, geometric_zscore], tick_labels=["Gaussian","powerlaw","geometric"])
plt.title("Box-plot of z-score Normalized Variables")
plt.show()

plt.hist( [B,Gaussian_zscore], bins=100, label=["original-Gaussian", "Normalized-Gaussian"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of gaussian variable")
plt.show()

plt.hist( [I,powerlaw_zscore], bins=100, label=["original-Powerlaw","Normalized-powerlaw"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of powerlaw variable")
plt.show()

plt.hist( [H,geometric_zscore], bins=100, label=["original-Geometric","Normalized-Geometric"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of geometric variable")
plt.show()


# # For each variable , convert the values in percentiles
def percentile(array):
    np.sort(array)
    curr_value = np.arange(1,len(array)+1)
    percentile_value = (curr_value/totalsize) * 100
    return percentile_value

def percentile_normalization(B, I, H):
    gaussian_percentile = percentile(B)
    powerlaw_percentile = percentile(I)
    geometric_percentile = percentile(H)
    return gaussian_percentile, powerlaw_percentile, geometric_percentile

gaussian_percentile, powerlaw_percentile, geometric_percentile = percentile_normalization(B, I, H)

plt.boxplot([gaussian_percentile, powerlaw_percentile, geometric_percentile], tick_labels=["Gaussian","powerlaw","geometric"])
plt.title("Box-plot of percentile normalized variables")
plt.show()

plt.hist( [B,gaussian_percentile], bins=100, label=["original-Gaussian", "Normalized-Gaussian"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of gaussian variable")
plt.show()

plt.hist( [I,powerlaw_percentile], bins=100, label=["original-Powerlaw","Normalized-powerlaw"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of powerlaw variable")
plt.show()

plt.hist( [H,geometric_zscore], bins=100, label=["original-Geometric","Normalized-Geometric"], edgecolor= ["gray","gray"])
plt.legend(loc = 'upper right')
plt.title("Histogram distribution of geometric variable")
plt.show()