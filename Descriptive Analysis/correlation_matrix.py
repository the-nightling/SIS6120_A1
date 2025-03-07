# Load data
import pandas as pd

wine_df: pd.DataFrame = pd.read_csv("winequality-red.csv", delimiter=";")

# Plot Correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix: pd.DataFrame = wine_df.corr()

plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    annot_kws={"size": 8},
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.title("Correlation Matrix Heatmap")
plt.show()
