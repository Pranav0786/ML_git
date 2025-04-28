import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

male_mean = 166
female_mean = 152
SD = 5

female_ht = pd.Series(np.random.normal(female_mean,SD,1000))
male_ht = pd.Series(np.random.normal(male_mean,SD,1000))

total = female_ht.size + male_ht.size

# plt.hist([female_ht, male_ht], bins = 100 , label = ['Female','Male'])
# plt.legend(loc = 'upper right')
# plt.show()

# top 50 female heights modified

female_ht[:50] = female_ht[:50] + 10
changed_mean = female_ht.mean()
print(changed_mean)

changed_SD = female_ht.std()
print(changed_SD)

plt.hist([female_ht, male_ht], bins = 100 , label = ['Female','Male'])
plt.legend(loc = 'upper right')
plt.show()