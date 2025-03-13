# Predicting-Battery-Remaining-Useful-Life-RUL-
Predicting the Remaining Useful Life (RUL) of batteries is crucial for ensuring reliability, safety, and efficiency in critical applications such as electric vehicles (EVs), aerospace, and energy storage systems. Unexpected battery failures can lead to operational disruptions and safety hazards, making accurate RUL prediction essential. In this study, we develop a machine learning-based approach to predict battery RUL using multiple regression models trained on a dataset containing features such as Discharge Time, Voltage Decrement, and Charging Time. We evaluate six models: Linear Regression, Random Forest, Extra Trees, Gradient Boosting, Neural Networks, and Support Vector Regression (SVR).
Found that Extra Trees model was performing the best-Using the tuned Extra Trees model, we predict RUL for 100 unseen battery samples and save the results. This study demonstrates the effectiveness of ensemble learning in battery RUL prediction, providing a robust tool for predictive maintenance and reliability assessment.


3 Different Data Uploaded
1️⃣ Main_Data.csv → The primary dataset used for training and testing the machine learning model.
2️⃣ Predict_RUL.csv → A dataset without RUL, where predictions need to be made using the trained model.
3️⃣ Predict_RUL_results.csv → The same as Predict_RUL.csv, but with an added Predicted RUL column from your model.



# Evaluating and Selecting the Best Performing Model Among Six Different Machine Learning Algorithms
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
# 🔹 Step 1: Load Battery Dataset
file_path = "/Users/estherchung/Desktop/Main_Data.csv"  # Update this with your dataset path
df = pd.read_csv(file_path)
# 🔹 Step 2: Define Features (X) and Target (y)
target_column = "RUL"  # Battery Remaining Useful Life
X = df.drop(columns=[target_column])  
y = df[target_column]
# 🔹 Step 3: Split Data into Training and Testing (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 🔹 Step 4: Define Multiple Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),  
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "SVR": SVR(kernel="rbf")
}
# 🔹 Step 5: Train, Predict & Evaluate Each Model
results = []

for model_name, model in models.items():
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save results
    results.append({"Model": model_name, "MSE": mse, "MAE": mae, "R² Score": r2})

    # Print results directly
    print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")

print(df.columns)
    


