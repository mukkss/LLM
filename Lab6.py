# linear_regression_boston.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset (CSV must be available in the same directory)
boston_df = pd.read_csv("boston_housing_data.csv")

print("Linear Regression on Boston Housing Dataset")

# Feature and target
X = boston_df[['RM']]  # Average number of rooms
y = boston_df['MEDV']  # Median house value

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Plotting
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Median House Price (MEDV)')
plt.title('Linear Regression: Boston Housing')
plt.legend()
plt.grid(True)
plt.show()






# polynomial_regression_auto_mpg.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Auto MPG dataset (CSV must be available in the same directory)
auto_df = pd.read_csv(" ")

print("Polynomial Regression on Auto MPG Dataset")

# Data Cleaning
auto_df['horsepower'] = auto_df['horsepower'].replace('?', np.nan).astype(float)
auto_df.dropna(subset=['horsepower'], inplace=True)

# Feature and target
X = auto_df[['horsepower']]
y = auto_df['mpg']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the Polynomial Regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predictions
y_pred = poly_model.predict(X_test_poly)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Plotting
plt.scatter(X_test, y_test, color='green', label='Actual')
# Sort values for a smooth curve
sorted_idx = X_test.squeeze().argsort()
plt.plot(X_test.iloc[sorted_idx], y_pred[sorted_idx], color='orange', linewidth=2, label='Predicted')
plt.xlabel('Horsepower')
plt.ylabel('MPG (Miles Per Gallon)')
plt.title('Polynomial Regression: Auto MPG')
plt.legend()
plt.grid(True)
plt.show()
