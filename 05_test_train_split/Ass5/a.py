import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm, zscore
import matplotlib.pyplot as plt

# Step 1: Generate Heights
def generate_heights(mean, sd, size):
    return np.random.normal(mean, sd, size)

np.random.seed(40)
male_heights = generate_heights(166, 5.5, 1000)
female_heights = generate_heights(152, 4.5, 1000)

# Step 2: Train-Test Split
male_train, male_test = train_test_split(male_heights, test_size=0.2, random_state=42)
female_train, female_test = train_test_split(female_heights, test_size=0.2, random_state=42)

train_data = np.concatenate([male_train, female_train])
train_labels = np.concatenate([np.ones(800), np.zeros(800)])

test_data = np.concatenate([male_test, female_test])
test_labels = np.concatenate([np.ones(200), np.zeros(200)])

# Step 3: Train Probability-Based Classifier
def classify(data, male_mean, male_sd, female_mean, female_sd):
    male_prob = norm.pdf(data, male_mean, male_sd)
    female_prob = norm.pdf(data, female_mean, female_sd)
    return (male_prob > female_prob).astype(int)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

male_mean, male_sd = np.mean(male_train), np.std(male_train)
female_mean, female_sd = np.mean(female_train), np.std(female_train)

train_preds = classify(train_data, male_mean, male_sd, female_mean, female_sd)
test_preds = classify(test_data, male_mean, male_sd, female_mean, female_sd)

print("Step 3: Classification Accuracy")
print(f'Train Accuracy: {accuracy(train_labels, train_preds):.2f}%')
print(f'Test Accuracy: {accuracy(test_labels, test_preds):.2f}%')

# Step 4: Impact of Outliers
female_train_sorted = np.sort(female_train)
female_train_sorted[-50:] += 10

num_outliers_added = 50

new_female_mean, new_female_sd = np.mean(female_train_sorted), np.std(female_train_sorted)
print("\nStep 4: Impact of Outliers")
print(f'New Female Mean: {new_female_mean:.2f}, New Female Std: {new_female_sd:.2f}')
print(f'Number of Outliers Added: {num_outliers_added}')

# Histogram before and after outliers
plt.figure(figsize=(10, 4))
plt.hist(female_train, bins=20, alpha=0.7, color='purple', edgecolor='black', label="Before Outliers")
plt.hist(female_train_sorted, bins=20, alpha=0.7, color='red', edgecolor='black', label="After Outliers")
plt.title("Female Heights - Outlier Injection")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Step 5: Remove Outliers using Z-score
female_train_zscores = zscore(female_train_sorted)
female_train_no_outliers = female_train_sorted[np.abs(female_train_zscores) < 3]

num_outliers_removed = len(female_train_sorted) - len(female_train_no_outliers)

final_female_mean, final_female_sd = np.mean(female_train_no_outliers), np.std(female_train_no_outliers)

print("\nStep 5: Removing Outliers")
print(f'New Female Mean (No Outliers): {final_female_mean:.2f}, New Female Std: {final_female_sd:.2f}')
print(f'Number of Outliers Removed: {num_outliers_removed}')

#
# Standard Deviation Impact Graph
std_values = [female_sd, new_female_sd, final_female_sd]
labels = ['Original', 'With Outliers', 'Outliers Removed']

plt.figure(figsize=(8, 5))
plt.bar(labels, std_values, color=['blue', 'red', 'green'])
plt.xlabel("Data State")
plt.ylabel("Standard Deviation")
plt.title("Impact of Outliers on Standard Deviation")
plt.show()

# Histogram before and after outlier removal
plt.figure(figsize=(10, 4))
plt.hist(female_train_sorted, bins=20, alpha=0.7, color='red', edgecolor='black', label="With Outliers")
plt.hist(female_train_no_outliers, bins=20, alpha=0.7, color='green', edgecolor='black', label="Outliers Removed")
plt.title("Female Heights - Outlier Removal")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# Standard Deviation Visualization Using Line Graph
plt.figure(figsize=(8, 5))

# X-axis categories
labels = ["Original", "With Outliers", "Outliers Removed"]
std_devs = [female_sd, new_female_sd, final_female_sd]

# Plot standard deviation changes
plt.plot(labels, std_devs, marker='o', linestyle='-', color='blue', markersize=8, linewidth=2, label="Standard Deviation")

# Labels & Title
plt.xlabel("Data State")
plt.ylabel("Standard Deviation (cm)")
plt.title("Impact of Outliers on Standard Deviation")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

plt.show()


## Step 6: Impact of Trimming (Fully Fixed)
k_values, train_accuracies, test_accuracies = [], [], []
print("\nStep 6: Impact of Trimming (Fully Fixed)")

for k in range(1, 26):
    # Compute trimming bounds
    lower_bound = np.percentile(female_train_no_outliers, k)
    upper_bound = np.percentile(female_train_no_outliers, 100 - k)
    
    # Trim female training data
    female_trimmed = female_train_no_outliers[(female_train_no_outliers >= lower_bound) & (female_train_no_outliers <= upper_bound)]
    
    # Rebuild training dataset with trimmed females
    new_train_data = np.concatenate([male_train, female_trimmed])
    new_train_labels = np.concatenate([np.ones(len(male_train)), np.zeros(len(female_trimmed))])  # 1 = Male, 0 = Female
    
    # Compute new mean & std for classification
    trimmed_mean, trimmed_sd = np.mean(female_trimmed), np.std(female_trimmed)

    # Classify using the updated trimmed data
    train_preds_trimmed = classify(new_train_data, male_mean, male_sd, trimmed_mean, trimmed_sd)
    test_preds_trimmed = classify(test_data, male_mean, male_sd, trimmed_mean, trimmed_sd)

    # Compute accuracy
    train_acc = accuracy(new_train_labels, train_preds_trimmed)
    test_acc = accuracy(test_labels, test_preds_trimmed)

    # Store results
    k_values.append(k)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f'Trimming {k}% - Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')

# Accuracy impact plot (Corrected)
plt.figure(figsize=(8, 5))
plt.plot(k_values, train_accuracies, label="Train Accuracy", marker='o', linestyle='-', color='blue')
plt.plot(k_values, test_accuracies, label="Test Accuracy", marker='s', linestyle='-', color='orange')
plt.xlabel("Trim Percentage (%)")
plt.ylabel("Accuracy")
plt.title("Impact of Trimming on Classifier Accuracy (Fully Fixed)")
plt.legend()
plt.grid(True)
plt.show()