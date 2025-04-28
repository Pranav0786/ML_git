import numpy as np
from scipy.stats import skewnorm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import time

np.random.seed(72)

class OptimizedQuantizationClassifier:
    def __init__(self, bin_size=1.0):
        self.bin_size = bin_size
        self.min_value = None
        self.bin_probs = {}  # Dictionary mapping bin index to probability
        self.global_prob = 0.5
    
    def _get_bin_index(self, value):
        """Convert value to bin index"""
        return int((value - self.min_value) / self.bin_size)
    
    def fit(self, X, y):
        # Calculate global probability
        self.global_prob = np.mean(y)
        
        # Find min/max values
        self.min_value = np.floor(np.min(X))
        max_value = np.ceil(np.max(X))
        
        # Create bins in a vectorized way
        bin_indices = np.floor((X - self.min_value) / self.bin_size).astype(int)
        
        # Use a fast approach to calculate probabilities for each bin
        unique_bins = np.unique(bin_indices)
        
        for bin_idx in unique_bins:
            mask = (bin_indices == bin_idx)
            if np.any(mask):
                self.bin_probs[bin_idx] = np.mean(y[mask])
    
    def predict(self, X):
       
        return self._predict_vectorized(X)
       
    
    def _predict_vectorized(self, X):
        
        bin_indices = np.floor((X - self.min_value) / self.bin_size).astype(int)
        
        # Initialize with global probability
        probs = np.full(len(X), self.global_prob)
        
        # Update probabilities based on bin membership
        for i, bin_idx in enumerate(bin_indices):
            if bin_idx in self.bin_probs:
                probs[i] = self.bin_probs[bin_idx]
        
        # Convert probabilities to predictions
        return (probs > 0.5).astype(int)
    
    
    def score(self, X, y):
        """Calculate accuracy score"""
        preds = self.predict(X)
        return np.mean(preds == y)


# 1. Generate 10000 female samples using skewnorm

female_heights = skewnorm.rvs(3, 150, 5, 10000)

# 2. Generate 10000 male samples using normal distribution
male_heights = np.random.normal(166, 5.5, 10000)

# Create labels (0 for female, 1 for male)
female_labels = np.zeros(10000)
male_labels = np.ones(10000)

# Combine the data
all_heights = np.concatenate([female_heights, male_heights])
all_labels = np.concatenate([female_labels, male_labels])

# 3. Use test-train split to set aside 20% for test data
X_train, X_test, y_train, y_test = train_test_split(
    all_heights, all_labels, test_size=0.2, random_state=52)

# Plot the height distributions
plt.figure(figsize=(10, 6))
plt.hist(female_heights, bins=50, label='Female')
plt.hist(male_heights, bins=50, label='Male')
plt.xlabel('Height (cm)')
plt.ylabel('Count')
plt.title('Height Distribution by Gender')
plt.legend()
plt.show()

# 4. Design a quantization/binning based classification algorithm

# 4.1 Try different bin sizes from 1cm to 10^-6cm
bin_sizes = [10**-i for i in range(0, 7)]  # 1cm to 0.000001cm

# 4.2 Shuffle and split the train data
# First shuffle
shuffle_idx = np.random.permutation(len(X_train))
X_shuffled, y_shuffled = X_train[shuffle_idx], y_train[shuffle_idx]

# Split into new train (6000) and validation (2000)
X_new_train, X_val = X_shuffled[:6000], X_shuffled[6000:8000]
y_new_train, y_val = y_shuffled[:6000], y_shuffled[6000:8000]

print(f"New training set size: {len(X_new_train)}")
print(f"Validation set size: {len(X_val)}")

# Store results
results = {'bin_size': [], 'train_acc': [], 'val_acc': [], 'train_time': []}

# 4.3 For each bin size, train and evaluate
for bin_size in bin_sizes:
    print(f"\nEvaluating bin size: {bin_size}")
    
    # 4.3.1 Train model on new train set
    start_time = time.time()
    model = OptimizedQuantizationClassifier(bin_size)
    model.fit(X_new_train, y_new_train)
    train_time = time.time() - start_time
    
    # 4.3.2 Measure accuracy on train and validation
    train_acc = model.score(X_new_train, y_new_train)
    val_acc = model.score(X_val, y_val)
    
    # Store results
    results['bin_size'].append(bin_size)
    results['train_acc'].append(train_acc)
    results['val_acc'].append(val_acc)
    results['train_time'].append(train_time)
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Training time: {train_time:.4f}s")

# 4.4 Determine most appropriate bin size
results_df = pd.DataFrame(results)
optimal_idx = results_df['val_acc'].idxmax()
optimal_bin_size = results_df.loc[optimal_idx, 'bin_size']

print("\n" + "="*50)
print(f"Optimal bin size: {optimal_bin_size} with validation accuracy: {results_df.loc[optimal_idx, 'val_acc']:.4f}")

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(results_df['bin_size'], results_df['train_acc'], marker='o', label='Train Accuracy')
plt.plot(results_df['bin_size'], results_df['val_acc'], marker='s', label='Validation Accuracy')
plt.xscale('log')
plt.xlabel('Bin Size (log scale)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Bin Size')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(results_df['bin_size'], results_df['train_time'], marker='o', label='Training Time')
plt.xscale('log')
plt.xlabel('Bin Size (log scale)')
plt.ylabel('Time (seconds)')
plt.title('Training Time vs. Bin Size')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 4.5 Train on full train set with optimal bin size and measure accuracy on test set
print("\n" + "="*50)
print(f"Training final model with optimal bin size: {optimal_bin_size}")

final_model = OptimizedQuantizationClassifier(optimal_bin_size)
final_model.fit(X_train, y_train)

# Evaluate on train and test sets
final_train_acc = final_model.score(X_train, y_train)
final_test_acc = final_model.score(X_test, y_test)

print(f"Final model train accuracy: {final_train_acc:.4f}")
print(f"Final model test accuracy: {final_test_acc:.4f}")

# Display detailed bin information for the final model
print("\nBin distribution in final model:")
print(f"Number of bins: {len(final_model.bin_probs)}")
