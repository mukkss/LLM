from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load and standardize data
data = load_iris()
X = StandardScaler().fit_transform(data.data)
y = data.target

# Apply PCA
X_pca = PCA(n_components=2).fit_transform(X)

# Custom colors
colors = ['brown', 'hotpink', 'purple']

# Plot with legend and custom colors
for i, label in enumerate(np.unique(y)):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                color=colors[i], label=data.target_names[label])

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Iris Dataset')
plt.legend()
plt.show()
