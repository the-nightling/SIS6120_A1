# Load data
import pandas as pd

wine_df: pd.DataFrame = pd.read_csv("winequality-red.csv", delimiter=";")

# Descriptive analysis
import matplotlib.pyplot as plt

descriptive_statistics_df: pd.DataFrame = wine_df.describe()
print(descriptive_statistics_df)

skewness = wine_df.skew()
print("Skewness:")
print(skewness)
