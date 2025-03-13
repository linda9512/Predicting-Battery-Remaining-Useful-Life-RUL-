# Evaluating and Comparing Model Performance Across Six Different Models
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load Battery Dataset 
file_path = "/Users/estherchung/Desktop/Main_Data.csv"
df = pd.read_csv(file_path)

# Define Features (X) and Target (y)
target_column = "RUL"  # Battery Remaining Useful Life
X = df.drop(columns=[target_column])  
y = df[target_column]

# Split Data into Training and Testing (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Multiple Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),  
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "SVR": SVR(kernel="rbf")
}

# Train, Predict & Evaluate Each Model
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save results
    results.append({"Model": model_name, "MSE": mse, "MAE": mae, "R² Score": r2})
    print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")






# Tune Hyperparameters for Random Forest, Gradient Boosting, and Extra Trees
param_grids = {
    "Random Forest": {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "Gradient Boosting": {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    },
    "Extra Trees": {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}

best_models = {}
for model_name, param_grid in param_grids.items():
    grid_search = GridSearchCV(
        eval(f"{model_name.replace(' ', '')}Regressor(random_state=42)"),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"\nBest {model_name} Parameters:", grid_search.best_params_)

# Evaluate Tuned Models
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{model_name} (Tuned) - MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")

# Save Best Models
os.makedirs("results/models/", exist_ok=True)
for model_name, model in best_models.items():
    joblib.dump(model, f"results/models/best_{model_name.replace(' ', '_').lower()}.pkl")





# Predict Unknown RUL Using the Best Model (Extra Trees)
model_path = "results/models/best_extra_trees.pkl"
best_et = joblib.load(model_path)

predict_file = "/Users/estherchung/Desktop/Predict_RUL.csv"
df_new = pd.read_csv(predict_file)

predicted_rul = best_et.predict(df_new)
df_new["Predicted RUL"] = predicted_rul

output_path = "/Users/estherchung/Desktop/predicted_RUL_results.csv"
df_new.to_csv(output_path, index=False)
print("\nPredicted RUL values saved successfully!")






# Create Correlation Matrix and Scatter Plots
features = ['Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)', 
            'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 
            'Charging time (s)']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    plt.scatter(df[feature], df["RUL"], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel("RUL")
    plt.title(f"RUL vs {feature}")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df[features + ["RUL"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of RUL and Features")
plt.show()
