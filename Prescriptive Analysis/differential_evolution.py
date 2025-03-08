# Load data
import pandas as pd

wine_df: pd.DataFrame = pd.read_csv("winequality-red.csv", delimiter=";")

# Separate features and target variable
X = wine_df.drop("quality", axis=1)
y = wine_df["quality"]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Standardize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=5)
model.fit(X_train_scaled, y_train)


# Define the objective function for optimization
def objective_function(params):
    params_df = pd.DataFrame([params], columns=X.columns)
    params_scaled = scaler.transform(params_df)
    prediction = model.predict(params_scaled)
    return -prediction  # Negate to maximize quality


# Define bounds for each feature based on the training data
bounds = [
    (X_train_scaled[:, i].min(), X_train_scaled[:, i].max())
    for i in range(X_train_scaled.shape[1])
]

# Perform optimization using Differential Evolution
from scipy.optimize import differential_evolution

result = differential_evolution(
    objective_function, bounds, strategy="best1bin", maxiter=1000, tol=1e-7
)
optimal_params = result.x
optimal_params_original_scale = scaler.inverse_transform([optimal_params])

# Create a dictionary of optimal parameters with column names
optimal_params_dict = {
    col: f"{param:.2f}"
    for col, param in zip(X.columns, optimal_params_original_scale[0])
}

# Print the optimal parameters with column names
print("Optimal parameters for maxmimum wine quality:")
for col, param in optimal_params_dict.items():
    print(f"{col}: {param}")
