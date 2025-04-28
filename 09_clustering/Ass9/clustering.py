import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('Cancer_Data.csv')
original_diagnosis = df['diagnosis']

df = df.drop(columns=['id', 'diagnosis'])  
df = df.dropna(axis=1, how='all')  
df = df.fillna(df.mean())  

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method - Inertia vs k')
plt.grid(True)
plt.show()

plt.plot(k_range, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis - Score vs k')
plt.grid(True)
plt.show()

point1 = np.array([k_range[0], inertia[0]])
point2 = np.array([k_range[-1], inertia[-1]])

distances = []
for i in range(len(k_range)):
    point = np.array([k_range[i], inertia[i]])
    numerator = np.abs(np.cross(point2 - point1, point1 - point))
    denominator = np.linalg.norm(point2 - point1)
    distance = numerator / denominator
    distances.append(distance)

optimal_k_inertia = k_range[np.argmax(distances)]

silhouette_differences = np.abs(np.diff(silhouette_scores))
max_diff_index = np.argmax(silhouette_differences)

optimal_k_silhouette = k_range[max_diff_index]   

optimal_k = optimal_k_inertia
print(f"Optimal number of clusters (based on inertia): {optimal_k}")
print(f"Optimal number of clusters (based on silhouette score): {optimal_k_silhouette}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(df_scaled)
labels = kmeans.labels_

diagnosis_numerical = original_diagnosis.map({'M': 1, 'B': 0})

homogeneity = homogeneity_score(diagnosis_numerical, labels)
print(f"Homogeneity Score for KMeans: {homogeneity:.4f}")

min_samples_for_eps = 5
neighbors = NearestNeighbors(n_neighbors=min_samples_for_eps)
neighbors_fit = neighbors.fit(df_scaled)
distances, indices = neighbors_fit.kneighbors(df_scaled)

k_distances = np.sort(distances[:, -1])  
plt.plot(k_distances)
plt.ylabel(f'{min_samples_for_eps}th Nearest Neighbor Distance')
plt.xlabel('Points sorted by distance')
plt.title('DBSCAN k-distance Graph (auto eps estimation)')
plt.grid(True)
plt.show()

eps_values = np.arange(2.2,4.2,0.1)
min_samples_values = range(2,8)

best_homogeneity = -1
best_eps = None
best_min_samples = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(df_scaled)
        
        if len(set(dbscan_labels)) > 1:  
            homogeneity_dbscan = homogeneity_score(diagnosis_numerical, dbscan_labels)
            if homogeneity_dbscan > best_homogeneity:
                best_homogeneity = homogeneity_dbscan
                best_eps = eps
                best_min_samples = min_samples


print(f"Best DBSCAN parameters: eps = {best_eps}, min_samples = {best_min_samples}")
print(f"Homogeneity Score for DBSCAN: {best_homogeneity:.4f}")