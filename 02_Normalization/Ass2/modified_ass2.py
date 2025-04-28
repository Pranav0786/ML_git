import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
from scipy.stats import rankdata


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


unique_gaussian = len(np.unique(B))
unique_powerlaw = len(np.unique(I))
unique_geometric = len(np.unique(H))

print(f"no. of unique values in Gaussian: {unique_gaussian}")
print(f"no. of unique values in Powerlaw: {unique_powerlaw}")
print(f"no. of unique values in Geometric: {unique_geometric}")


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


# Percentile Normalization - tie breaking methods: 
    
def percentile(array, method_used):
    curr_value = rankdata(array, method=method_used)
    percentile_value = ((curr_value-1) / (totalsize-1)) * 100
    return percentile_value

tie_methods = ["average","min","max","ordinal","dense"]


for method in tie_methods:
    gaussian_percentile = percentile(B, method)
    powerlaw_percentile = percentile(I, method)
    geometric_percentile = percentile(H, method)

    plt.boxplot([gaussian_percentile, powerlaw_percentile, geometric_percentile],
                labels=["Gaussian", "Powerlaw", "Geometric"],
                patch_artist=True,
                boxprops=dict(facecolor="lightgreen"))
    plt.title(f"Box-plot of Percentile Normalized Variables ({method} method)")
    plt.show()

    plot_histogram([B, gaussian_percentile],["Original-Gaussian", f"Normalized-Gaussian ({method})"],f"Histogram - Gaussian variable using {method}")

    plot_histogram([I, powerlaw_percentile],["Original-Powerlaw", f"Normalized-Powerlaw ({method})"],f"Histogram - Powerlaw variable using {method}")

    plot_histogram([H, geometric_percentile],["Original-Geometric", f"Normalized-Geometric ({method})"],f"Histogram - Geometric variable using {method}")

geo_percentile = percentile(H, "dense")
plt.hist(geo_percentile, bins=100, label="fv", alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("dense hist")
plt.legend(loc='upper right')
plt.show()
