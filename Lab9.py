from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load and scale data
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y_true = data.target

# Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=0)
y_pred = kmeans.fit_predict(X)

# Evaluation
print("Silhouette Score:", silhouette_score(X, y_pred))
print("Adjusted Rand Index:", adjusted_rand_score(y_true, y_pred))

# PCA for 2D visualization
X_pca = PCA(n_components=2).fit_transform(X)

# Plot clustering result
plt.figure(figsize=(8,4))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred,
                palette="coolwarm", s=60)
plt.title("K-Means Clustering")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# Plot actual labels
plt.figure(figsize=(8,4))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_true,
                palette="Set2", s=60)
plt.title("Actual Labels")
plt.legend(title="Actual Class")
plt.grid(True)
plt.show()
