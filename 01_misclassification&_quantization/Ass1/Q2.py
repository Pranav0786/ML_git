import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

female_ht = pd.Series(np.random.normal(152,5,1000))
male_ht = pd.Series(np.random.normal(166,5,1000))

total = female_ht.size + male_ht.size

plt.hist([female_ht, male_ht], bins = 100 , label = ['Female','Male'])
plt.legend(loc = 'upper right')
plt.show()


#calculating probability 

F_mean = female_ht.mean()
M_mean = male_ht.mean()

F_SD = female_ht.std()
M_SD = male_ht.std()

misclassifiedmale = 0


for curr_ht in male_ht:
    female_prob = norm.pdf(curr_ht,F_mean,F_SD)
    male_prob = norm.pdf(curr_ht,M_mean,M_SD)
    if male_prob < female_prob:
        misclassifiedmale+=1
        # print("misplaced male ", curr_ht)

misclassifiedfemale = 0

for curr_ht in female_ht:
    female_prob = norm.pdf(curr_ht,F_mean,F_SD)
    male_prob = norm.pdf(curr_ht,M_mean,M_SD)
    if male_prob > female_prob:
        misclassifiedfemale+=1
        # print("misplaced female ", curr_ht)

print("MisplacedMales & misplacedFemales count : ", misclassifiedmale,misclassifiedfemale)
total_misclassification = misclassifiedfemale + misclassifiedmale
rate = (total_misclassification/total) * 100
print("Misclassification rate : ", rate)


#Deriving threshold height to seperate male and female
max_female_height = int(female_ht.max())
min_male_height = int(male_ht.min())

threshold = min_male_height
min_misclassification = 10**15

for curr_height in range( min_male_height, max_female_height + 1 ):
    misclassification = 0
    misclassification += sum(male_ht < curr_height) + sum(female_ht > curr_height)

    if(misclassification < min_misclassification ):
        min_misclassification = misclassification
        threshold = curr_height

print("Threshold Height: ", threshold)




#quantize the  data at scale of 0.5 cm and empirically estimate the likelihood of male female in each segment based on majority 

def frequency(scale, height):
    quantized_val = np.floor(height/scale)
    no_of_ppl = quantized_val.value_counts()
    return no_of_ppl

new_female_ht = frequency(0.8, female_ht)
new_male_ht = frequency(0.8, male_ht)

min_overlap_height = int(new_male_ht.index.min())
max_overlap_height = int(new_female_ht.index.max())

total_misplaced = 0
for ht in range(min_overlap_height, max_overlap_height+1):
    # ht = float(ht)  
    Female_count = new_female_ht.get(ht, 0)  # Use .get() for safe access
    Male_count = new_male_ht.get(ht, 0)

    misplaced = min(Female_count,Male_count)
    total_misplaced += misplaced

print("Total Misplaced: ", total_misplaced)

misplaced_rate = (total_misplaced/total)*100
print("rate of misplaced : ", misplaced_rate)

    
