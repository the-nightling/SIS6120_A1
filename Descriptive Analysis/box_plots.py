# Load data
import pandas as pd

wine_df: pd.DataFrame = pd.read_csv("winequality-red.csv", delimiter=";")

# Plot Box and Whisker
import matplotlib.pyplot as plt

columns = wine_df.columns
fig, axes = plt.subplots(4, 3, figsize=(15, 12))

for i, ax in enumerate(axes.flatten()):
    if i < len(columns):
        wine_df.boxplot(column=columns[i], ax=ax)
        ax.set_title(columns[i])
        ax.axes.get_xaxis().set_visible(False)
    else:
        ax.axis("off")

plt.tight_layout()
plt.show()
