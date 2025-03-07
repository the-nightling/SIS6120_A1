# Load data
import pandas as pd

wine_df: pd.DataFrame = pd.read_csv("winequality-red.csv", delimiter=";")

# Plot Histograms
import matplotlib.pyplot as plt

columns = wine_df.columns
fig, axes = plt.subplots(4, 3, figsize=(15, 12))

for i, ax in enumerate(axes.flatten()):
    if i < len(columns):
        wine_df[columns[i]].hist(ax=ax, bins=20)
        ax.set_title(columns[i])
    else:
        ax.axis("off")

plt.tight_layout()
plt.show()
