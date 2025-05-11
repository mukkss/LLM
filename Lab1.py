|import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/content/ibm_hr_employee-attrition.csv")
print("Dataset loaded.")

# Display numerical and categorical columns
num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(exclude='number').columns
print(f"Numerical columns: {list(num_cols)}")
print(f"Categorical columns: {list(cat_cols)}")

# Numerical column analysis
col = input("Enter numerical column: ").strip()
if col in df.columns:
    x = df[col].dropna()
    print(f"\nMean: {x.mean()}\nMedian: {x.median()}\nMode: {x.mode().iloc[0]}")
    print(f"Standard Deviation: {x.std()}\nVariance: {x.var()}\nRange: {x.max() - x.min()}")

    # Generate histogram and boxplot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(x, kde=True, bins=10, color='blue')
    plt.title(f"Histogram of {col}")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=x, color='lightgreen')
    plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

    # Outlier detection using IQR
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = x[(x < lower_bound) | (x > upper_bound)]

    print(f"\nNumber of outliers in {col}: {len(outliers)}")
    print(outliers)
else:
    print("Invalid numerical column.")

# Categorical column analysis
cat = input("\nEnter categorical column: ").strip()
if cat in df.columns:
    counts = df[cat].value_counts()
    chart = input("Chart type (bar/pie): ").lower().strip()

    if chart == 'bar':
        counts.plot(kind='bar', color='orange')
        plt.title(f"Bar Chart of {cat}")
        plt.xlabel(cat)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    elif chart == 'pie':
        counts.plot(kind='pie', autopct='%.1f%%', figsize=(5, 5))
        plt.title(f"Pie Chart of {cat}")
        plt.ylabel('')
        plt.tight_layout()
        plt.show()

    else:
        print("Invalid chart type. Please choose 'bar' or 'pie'.")

    print(f"\nCategory Frequencies for {cat}:\n{counts}")
else:
    print("Invalid categorical column.")
