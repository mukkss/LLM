from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to test k-NN (weighted or not)
def test_knn(k_vals, weighted=False):
    w = 'distance' if weighted else 'uniform'
    for k in k_vals:
        model = KNeighborsClassifier(n_neighbors=k, weights=w).fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"k={k} | Acc={accuracy_score(y_test, pred):.2f} | F1={f1_score(y_test, pred, average='weighted'):.2f}")

# Run for k = 1, 3, 5
print("Regular k-NN:"); test_knn([1, 3, 5])
print("\nWeighted k-NN:"); test_knn([1, 3, 5], weighted=True)
