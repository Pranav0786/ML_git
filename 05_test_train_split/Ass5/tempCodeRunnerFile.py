def trim_height_data(heights, k):
    lower_bound = np.percentile(heights, k)
    upper_bound = np.percentile(heights, 100-k)
    return heights[(heights >= lower_bound) & (heights <= upper_bound)]

train_accuracy_rate = []
test_accuracy_rate = []
kval = np.arange(1,26,1)

for k in np.arange(1, 26, 1):
    trimmed_female_train = trim_height_data( female_train , k)

    trimmed_female_mean = trimmed_female_train.mean()
    trimmed_female_SD = trimmed_female_train.std()

    train_accuracy_male_trimmed = accuracy(male_train, "male", train_male_mean, train_male_SD, trimmed_female_mean, trimmed_female_SD)
    male_frac1 = len(male_train)/ (len(male_train)+len(female_train))

    train_accuracy_female_trimmed = accuracy(trimmed_female_train, "female", train_male_mean, train_male_SD, trimmed_female_mean, trimmed_female_SD)
    total_train_accuracy_trimmed = ( male_frac1 * train_accuracy_male_trimmed + (1-male_frac1)*train_accuracy_female_trimmed)*100
    train_accuracy_rate.append(total_train_accuracy_trimmed)
    

    test_accuracy_male_trimmed = accuracy(male_test, "male", train_male_mean, train_male_SD, trimmed_female_mean, trimmed_female_SD)
    male_frac = len(male_test)/ (len(male_test)+len(female_test))

    test_accuracy_female_trimmed = accuracy(female_test, "female", train_male_mean, train_male_SD, trimmed_female_mean, trimmed_female_SD)
    total_test_accuracy_trimmed = ( male_frac* test_accuracy_male_trimmed + (1-male_frac)* test_accuracy_female_trimmed) * 100
    test_accuracy_rate.append(total_test_accuracy_trimmed)

    print(f"k={k}%, Train acc: {total_train_accuracy_trimmed:.4f}, Test acc: {total_test_accuracy_trimmed:.4f}")


plt.plot(["original", "with outliers", "after"], 
         [female_train.std(), modified_female_train.std(), filtered_female_train.std()], 
         marker='o', linestyle='-', color='red')

plt.xlabel("Stages")
plt.ylabel("SD")
plt.title("Change in SD of female hts")
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(kval, train_accuracy_rate, marker='o', linestyle='-', color='blue', label="Train accuracy Rate")
plt.plot(kval, test_accuracy_rate, marker='s', linestyle='-', color='red', label="Test accuracy Rate")

plt.xlabel("Trim Percentage (%)")
plt.ylabel("accuracy Rate (%)")
plt.title("Impact of Trimming on accuracy Rate")
plt.legend()
plt.grid(True)
plt.show()


