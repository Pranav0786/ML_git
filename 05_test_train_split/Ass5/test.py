import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm, zscore
import matplotlib.pyplot as plt

np.random.seed(60)

male_mean = 166
female_mean = 152
SD_male = 5.5
SD_female = 4.5


female_ht = pd.Series(np.random.normal(female_mean, SD_female, 1000))
male_ht = pd.Series(np.random.normal(male_mean, SD_male, 1000))


female_train, female_test = train_test_split(female_ht, test_size=0.2, random_state=0)
male_train, male_test = train_test_split(male_ht, test_size=0.2, random_state=0)


train_male_mean = male_train.mean()
train_female_mean = female_train.mean()
train_male_SD = male_train.std()
train_female_SD = female_train.std()


# Probability-based classification
def classification(height, male_mean, male_SD, female_mean, female_SD):
    male_prob = norm.pdf(height, male_mean, male_SD)
    female_prob = norm.pdf(height, female_mean, female_SD)
    return "male" if male_prob > female_prob else "female"


def accuracy(heights, gender, male_mean, male_SD, female_mean, female_SD):
    correct_predictions = sum(
        classification(ht, male_mean, male_SD, female_mean, female_SD) == gender
        for ht in heights
    )
    return correct_predictions

male_train_accuracy = accuracy(male_train, "male", train_male_mean, train_male_SD, train_female_mean, train_female_SD)
female_train_accuracy = accuracy(female_train, "female", train_male_mean, train_male_SD, train_female_mean, train_female_SD)
total_train_accuracy = (male_train_accuracy + female_train_accuracy) / 2 * 100

print("Training data accuracy male:", male_train_accuracy)
print("Training data accuracy female:", female_train_accuracy)
print("Training data accuracy total:", total_train_accuracy)

male_test_accuracy = accuracy(male_test, "male", train_male_mean, train_male_SD, train_female_mean, train_female_SD)
female_test_accuracy = accuracy(female_test, "female", train_male_mean, train_male_SD, train_female_mean, train_female_SD)
total_test_accuracy = (male_test_accuracy + female_test_accuracy) / 2 * 100

# print("Testing data accuracy male:", male_test_accuracy)
# print("Testing data accuracy female:", female_test_accuracy)
print("Testing data accuracy total:", total_test_accuracy)

# Impact of outliers
top_heights= sorted(female_train)[-50:]
top_female = [] 
for i in top_heights:
    top_female.append(i+10)
modified_female_train = pd.Series(sorted(female_train[:750]) + top_female)


# Train the probability-based classification algorithm on this altered train data 
# 1. Estimate the classification accuracy on both the train and test data 

mod_female_train_mean = modified_female_train.mean()
mod_female_train_SD = modified_female_train.std()

print(" mod_female_train_mean",mod_female_train_mean)
print(" mod_female_train_SD", mod_female_train_SD)

modified_female_train_accuracy = accuracy(modified_female_train, "female", train_male_mean, train_male_SD, mod_female_train_mean, mod_female_train_SD)
# print("modified female train accuracy:", modified_female_train_accuracy)
modified_male_train_accuracy = accuracy(male_train, "male", train_male_mean, train_male_SD, mod_female_train_mean, mod_female_train_SD)
# print("modified male train accuracy:", modified_male_train_accuracy)

total_mod_train_accuracy = (modified_female_train_accuracy + modified_male_train_accuracy)/2 * 100
print("modified total training data accuracy : ", total_mod_train_accuracy)


modified_female_test_accuracy = accuracy(female_test, "female", train_male_mean, train_male_SD, mod_female_train_mean, mod_female_train_SD)
# print("modified female train accuracy:", modified_female_test_accuracy)
modified_male_test_accuracy = accuracy(male_test, "male", train_male_mean, train_male_SD, mod_female_train_mean, mod_female_train_SD)
# print("modified male train accuracy:", modified_male_test_accuracy)

total_mod_test_accuracy = (modified_female_test_accuracy + modified_male_test_accuracy)/2 * 100
print("modified total testing data accuracy: ", total_mod_test_accuracy)


# Remove outliers from the train data using z-score method on female data 
female_zscore_train = zscore(modified_female_train)
cutoff = 3
filtered_female_train = modified_female_train[abs(female_zscore_train) <= cutoff]

num_outliers_removed = len(modified_female_train) - len(filtered_female_train)
print("Outliers removed based on cutoff 2:", num_outliers_removed)

filtered_female_mean = filtered_female_train.mean()
filtered_female_SD = filtered_female_train.std()

male_train_accuracy_after = accuracy(male_train, "male", train_male_mean, train_male_SD, filtered_female_mean, filtered_female_SD)
female_train_accuracy_after = accuracy(filtered_female_train, "female", train_male_mean, train_male_SD, filtered_female_mean, filtered_female_SD)
total_train_accuracy_after = (male_train_accuracy_after + female_train_accuracy_after) / 2 * 100

# print("Training data accuracy male (after outlier removal):", male_train_accuracy_after)
# print("Training data accuracy female (after outlier removal):", female_train_accuracy_after)
print("Training data accuracy total (after outlier removal):", total_train_accuracy_after)



male_test_accuracy_after = accuracy(male_test, "male", train_male_mean, train_male_SD, filtered_female_mean, filtered_female_SD)
female_test_accuracy_after = accuracy(female_test, "female", train_male_mean, train_male_SD, filtered_female_mean, filtered_female_SD)
total_test_accuracy_after = (male_test_accuracy_after + female_test_accuracy_after) / 2 * 100

# print("Testing data accuracy male (after outlier removal):", male_test_accuracy_after)
# print("Testing data accuracy female (after outlier removal):", female_test_accuracy_after)
print("Testing data accuracy total (after outlier removal):", total_test_accuracy_after)


# Impact of Trimming 

def trim_height_data(heights, k):
    lower_bound = np.percentile(heights, k)
    upper_bound = np.percentile(heights, 100-k)
    return heights[(heights >= lower_bound) & (heights <= upper_bound)]

train_accuracy_rate = []
test_accuracy_rate = []
kval = range(1,26)

for k in kval :
    trimmed_female_train = trim_height_data( modified_female_train , k)
    flen = len(trimmed_female_train)

    trimmed_female_mean = trimmed_female_train.mean()
    trimmed_female_SD = trimmed_female_train.std()

    train_accuracy_male_trimmed = accuracy(male_train, "male", train_male_mean, train_male_SD, trimmed_female_mean, trimmed_female_SD)

    train_accuracy_female_trimmed = accuracy(trimmed_female_train, "female", train_male_mean, train_male_SD, trimmed_female_mean, trimmed_female_SD)
    total_train_accuracy_trimmed = ((train_accuracy_male_trimmed + train_accuracy_female_trimmed) )/(flen+ len(male_train))
    train_accuracy_rate.append(total_train_accuracy_trimmed)
    

    test_accuracy_male_trimmed = accuracy(male_test, "male", train_male_mean, train_male_SD, trimmed_female_mean, trimmed_female_SD)

    test_accuracy_female_trimmed = accuracy(female_test, "female", train_male_mean, train_male_SD, trimmed_female_mean, trimmed_female_SD)
    total_test_accuracy_trimmed = ((test_accuracy_male_trimmed + test_accuracy_female_trimmed) ) /(len(male_test) + len(female_test))
    test_accuracy_rate.append(total_test_accuracy_trimmed)

    print(f"k={k}%, Train acc: {total_train_accuracy_trimmed:.4f}, Test acc: {total_test_accuracy_trimmed:.4f}")




plt.figure(figsize=(8, 5))
plt.plot(kval, train_accuracy_rate, marker='o', linestyle='-', color='blue', label="Train accuracy Rate")
plt.plot(kval, test_accuracy_rate, marker='s', linestyle='-', color='red', label="Test accuracy Rate")

plt.xlabel("Trim Percentage (%)")
plt.ylabel("accuracy Rate (%)")
plt.title("Impact of Trimming on accuracy Rate")
plt.legend()
plt.grid(True)
plt.show()


