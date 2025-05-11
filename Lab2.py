import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
data = sns.load_dataset('iris')

# Select two numerical columns
x = 'sepal_length'
y = 'petal_length'

# Scatter plot
plt.scatter(data[x], data[y])
plt.xlabel(x)
plt.ylabel(y)
plt.title(f"Scatter plot: {x} vs {y}")
plt.show()

# Pearson correlation coefficient
corr = data[[x, y]].corr().iloc[0, 1]
print(f"Pearson Correlation Coefficient between {x} and {y}: {corr:.2f}")

# Covariance matrix
print("\nCovariance Matrix:")
print(data[[x, y]].cov())

# Full correlation matrix and heatmap
corr_matrix = data.select_dtypes(include='number').corr()

print("\nFull Correlation Matrix:")
print(corr_matrix)

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()
