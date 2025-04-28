import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm,zscore

male_mean = 166
female_mean = 152
SD = 5

female_ht = pd.Series(np.random.normal(female_mean,SD,1000))
male_ht = pd.Series(np.random.normal(male_mean,SD,1000))

total = female_ht.size + male_ht.size

plt.boxplot(female_ht,  patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("original female heights")
plt.show()

plt.hist([female_ht, male_ht], bins = 100 , label = ['Female','Male'])
plt.title("original distribution")
plt.legend(loc = 'upper right')
plt.show()

# top 50 female heights modified

top_indices = np.argsort(female_ht)[-50:]
female_ht[top_indices] += 10

changed_mean = female_ht.mean()
print(changed_mean)

changed_SD = female_ht.std()
print(changed_SD)

plt.hist([female_ht, male_ht], bins = 100 , label = ['Female','Male'])
plt.title("distribution after changing female heights")
plt.legend(loc = 'upper right')
plt.show()

#calculating probability 

F_mean = female_ht.mean()
M_mean = male_ht.mean()

F_SD = female_ht.std()
M_SD = male_ht.std()

def classification_cal(female_ht, male_ht, F_mean, F_SD, M_mean, M_SD, total):
    misclassified_male = 0
    for curr_ht in male_ht:
        female_prob = norm.pdf(curr_ht, F_mean, F_SD)
        male_prob = norm.pdf(curr_ht, M_mean, M_SD)
        if male_prob < female_prob:
            misclassified_male += 1

    misclassified_female = 0
    for curr_ht in female_ht:
        female_prob = norm.pdf(curr_ht, F_mean, F_SD)
        male_prob = norm.pdf(curr_ht, M_mean, M_SD)
        if male_prob > female_prob:
            misclassified_female += 1

    total_misclassification = misclassified_female + misclassified_male
    misclassification_rate = (total_misclassification / total) * 100

    # print("Misplaced Males & Misplaced Females count:", misclassified_male, misclassified_female)
    # print("Misclassification rate:", misclassification_rate)

    return misclassification_rate

mis_rate = classification_cal(
    female_ht, male_ht, F_mean, F_SD, M_mean, M_SD, total
)
print("misclassification rate: ", mis_rate)


#Deriving threshold height to seperate male and female

max_female_height = float(female_ht.max())
min_male_height = float(male_ht.min())

threshold = min_male_height
min_misclassification = 10**12

for curr_height in np.arange(min_male_height, max_female_height + 1.0, 1.0):  
    misclassification = 0
    misclassification += sum(male_ht < curr_height) + sum(female_ht > curr_height)

    if(misclassification < min_misclassification ):
        min_misclassification = misclassification
        threshold = curr_height

print("Threshold Height: ", threshold)


#box and whisker plot. User of whiskers to find outliers
plt.boxplot(female_ht,  patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("modified female height")
plt.show()

#Parametric –  
#1. convert heights into z score

Female_ht_zscore  = zscore(female_ht)

#experiment with z score cutoffs such as 2 and 3 ( on both sides) 
outliers_with_cutoff2 = 0
outliers_with_cutoff3 = 0
for curr_ht in Female_ht_zscore:
    if curr_ht > 2 or curr_ht < -2:
        outliers_with_cutoff2 += 1
    if curr_ht >3 or curr_ht < -3:
        outliers_with_cutoff3 += 1
print("Outliers based on cutoff 2: ",outliers_with_cutoff2)
print("Outliers based on cutoff 3: ", outliers_with_cutoff3)


# Detection and removal based on inter quartile range 

Q1 = female_ht.quantile(0.25)
Q3 = female_ht.quantile(0.75)
IQR = Q3 - Q1

upper_bound = Q3 + 1.5*IQR
lower_bound = Q1 - 1.5*IQR

IQR_outliers = []
for curr_height in female_ht:
    if curr_height < lower_bound or curr_height > upper_bound:
        IQR_outliers.append(curr_height)

print("IQR_Outliers length",len(IQR_outliers))


# Detection of Outliers Based on Median Absolute Deviation (MAD)

median = female_ht.median()
absolute_diff = pd.Series(abs(female_ht - median))
MAD = absolute_diff.median()

MAD_cutoff = 3

modified_zscore = 0.6745 * (female_ht - median) / MAD

MAD_outliers = []
for i in range(len(female_ht)):  
    if modified_zscore[i] > MAD_cutoff or modified_zscore[i] < -MAD_cutoff:  
        MAD_outliers.append(female_ht[i])

print("MAD_Outliers length",len(MAD_outliers))


# Remove data labelled as outliers using z score or iqr or MAD cutoffs 

without_IQR_outliers = female_ht[(female_ht >= lower_bound) & (female_ht <= upper_bound)]
without_MAD_outliers = female_ht[abs(modified_zscore) <= MAD_cutoff]



plt.boxplot([female_ht, without_IQR_outliers, without_MAD_outliers], 
            patch_artist=True, boxprops=dict(facecolor="lightblue"))

plt.xticks([1, 2, 3], ["Original", "IQR Filtered", "MAD Filtered"])
plt.title("Comparison of Outlier Removal Methods using Boxplots")
plt.show()



# Data trimming- drop lower and upper k% data(vary k between 1% to 15% in increments of 1%) from 1.a and run classification algorithms. Observe impact on accuracy via scatter plot 

def trim_height_data(heights, k):
    lower_drop = np.percentile(heights, k)
    upper_drop = np.percentile(heights, 100-k)
    return heights[(heights >= lower_drop) & (heights <= upper_drop)]

misclassification_rate = []

kval = np.arange(1,16,1)

for k in np.arange(1,16, 1):
    trimmed_female = trim_height_data(female_ht, k)
    trimmed_male = trim_height_data(male_ht, k)

    total_data = trimmed_female.size + trimmed_male.size
    modified_Fmean = trimmed_female.mean()
    modified_Mmean = trimmed_male.mean()
    modified_FSD = trimmed_female.std()
    modified_MSD = trimmed_male.std()

    misclassification_rate.append( classification_cal(trimmed_female, trimmed_male, modified_Fmean, modified_FSD, modified_Mmean, modified_MSD, total_data))

plt.scatter(kval, misclassification_rate, color = 'green')
plt.xlabel("trim percentage %")
plt.ylabel("misclassification rate %")
plt.show()

