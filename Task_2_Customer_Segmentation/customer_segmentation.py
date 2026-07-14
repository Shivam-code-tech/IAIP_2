import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
df = pd.read_csv("Mall_Customers.csv")
print("--- Dataset Preview ---")
print(df.head())

# 2. Feature Selection
# Extracting Annual Income (Column index 3) and Spending Score (Column index 4)
X = df.iloc[:, [3, 4]].values

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow Method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method chart
plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker="o", linestyle="--", color="red")
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
os.makedirs("output", exist_ok=True)
plt.savefig("output/elbow_method.png")
plt.close()  # Close plot to prepare for the final segment chart

# 5. Applying K-Means with optimal clusters (k=5)
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# 6. Visualizing the Clusters
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")

# Plot each cluster group with clear business profiles
colors = ["blue", "green", "red", "cyan", "magenta"]
labels = [
    "Cluster 1 (High Income, Low Spend)",
    "Cluster 2 (Average Income, Average Spend)",
    "Cluster 3 (High Income, High Spend - Target)",
    "Cluster 4 (Low Income, High Spend)",
    "Cluster 5 (Low Income, Low Spend)",
]

for i in range(5):
    plt.scatter(
        X[y_kmeans == i, 0],
        X[y_kmeans == i, 1],
        s=80,
        c=colors[i],
        label=labels[i],
        edgecolor="k",
        alpha=0.7,
    )

# Plotting cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=250,
    c="yellow",
    marker="X",
    edgecolor="black",
    label="Centroids",
)

plt.title("Customer Segments (K-Means Clustering)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(loc="best")
plt.tight_layout()

# Save visual layout
plt.savefig("output/customer_segments.png")
print("\nSuccess! Output images saved inside the 'output/' folder.")
plt.show()