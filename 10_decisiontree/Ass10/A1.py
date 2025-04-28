# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# 1. Load Dataset
df = pd.read_csv('Cancer_Data.csv')

# 2. Drop 'Id' column
df = df.drop(columns=['id'])

# 3. Define Features and Target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# 4. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Skew the training data
X_train_M = X_train[y_train == 'M']
y_train_M = y_train[y_train == 'M']

X_train_B = X_train[y_train == 'B']
y_train_B = y_train[y_train == 'B']

X_moved, y_moved = resample(X_train_M, y_train_M, replace=False, n_samples=120, random_state=42)

X_train = X_train.drop(X_moved.index)
y_train = y_train.drop(y_moved.index)

X_test = pd.concat([X_test, X_moved], axis=0)
y_test = pd.concat([y_test, y_moved], axis=0)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print(f"Train Shape after skewing: {X_train.shape}, {y_train.shape}")
print(f"Test Shape after skewing: {X_test.shape}, {y_test.shape}")

# 6. Build 10 Decision Trees with feature and sample bagging
trees = []
tree_accuracies = []

for i in range(10):
    X_sample, y_sample = resample(X_train, y_train, replace=True, random_state=i)
    tree = DecisionTreeClassifier(max_features='sqrt', class_weight='balanced', random_state=i)
    tree.fit(X_sample, y_sample)
    trees.append(tree)
    tree_accuracies.append(tree.score(X_sample, y_sample))

print("Training accuracies of 10 initial trees:", tree_accuracies)

# 7. Combine Feature Importances (average)
importances = np.zeros(X_train.shape[1])

for tree in trees:
    importances += tree.feature_importances_

importances /= len(trees)

feature_importance_dict = dict(zip(X_train.columns, importances))
#print(f"feature importance: ", feature_importance_dict)
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
selected_features = [feat for feat, imp in sorted_features[:10]]
print(f"Selected Features: {selected_features}")

# 8. Train 10 New Trees with Selected Features
trees_selected = []
tree_selected_accuracies = []

for i in range(10):
    X_sample, y_sample = resample(X_train[selected_features], y_train, replace=True, random_state=100+i)
    tree = DecisionTreeClassifier(max_features='sqrt', class_weight='balanced', random_state=100+i)
    tree.fit(X_sample, y_sample)
    trees_selected.append(tree)
    tree_selected_accuracies.append(tree.score(X_sample, y_sample))

print("Train accuracies of 10 base trees after feature selection:", tree_selected_accuracies)
print(f"Average Train Accuracy of 10 base trees: {np.mean(tree_selected_accuracies):.4f}")

# 8 Evaluate Base Trees on Test Data
tree_selected_test_accuracies = []

for tree in trees_selected:
    test_acc = tree.score(X_test[selected_features], y_test)
    tree_selected_test_accuracies.append(test_acc)

print("Test accuracies of 10 base trees:\n", tree_selected_test_accuracies)
print(f"\nAverage Test Accuracy of 10 base trees: {np.mean(tree_selected_test_accuracies):.4f}")

# 9. Prepare Meta-Features predictions of individual trees as extra features.


tree_preds_train = []
for tree in trees_selected:
    preds = tree.predict(X_train[selected_features])
    preds_num = np.where(preds == 'M', 1, 0)
    tree_preds_train.append(preds_num)

tree_preds_train = np.array(tree_preds_train).T
X_meta_train = np.hstack((X_train[selected_features].values, tree_preds_train))
y_meta_train = y_train.copy()

tree_preds_test = []
for tree in trees_selected:
    preds = tree.predict(X_test[selected_features])
    preds_num = np.where(preds == 'M', 1, 0)
    tree_preds_test.append(preds_num)

tree_preds_test = np.array(tree_preds_test).T
X_meta_test = np.hstack((X_test[selected_features].values, tree_preds_test))
y_meta_test = y_test.copy()

# 10. Encode Labels for Logistic Regression
y_meta_train_num = y_meta_train.map({'M': 1, 'B': 0})
y_meta_test_num = y_meta_test.map({'M': 1, 'B': 0})

# 11. Train Final Classifiers
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_meta_train, y_meta_train_num)

master_tree = DecisionTreeClassifier(class_weight='balanced', random_state=999)
master_tree.fit(X_meta_train, y_meta_train)

# 12. Evaluate Models
y_pred_log_reg = log_reg.predict(X_meta_test)
acc_log_reg = accuracy_score(y_meta_test_num, y_pred_log_reg)
recall_log_reg = recall_score(y_meta_test_num, y_pred_log_reg, pos_label=1)

y_pred_master_tree = master_tree.predict(X_meta_test)
acc_master_tree = accuracy_score(y_meta_test, y_pred_master_tree)
recall_master_tree = recall_score(y_meta_test, y_pred_master_tree, pos_label='M')

print("\nModel Performances on Test Data:")
print(f"Logistic Regression -> Accuracy: {acc_log_reg:.4f}, Recall (M class): {recall_log_reg:.4f}")
print(f"Master Decision Tree -> Accuracy: {acc_master_tree:.4f}, Recall (M class): {recall_master_tree:.4f}")
