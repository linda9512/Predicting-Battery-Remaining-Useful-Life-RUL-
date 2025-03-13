#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
    


# In[9]:


import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 🔹 Step 1: Load Battery Dataset
file_path = "/Users/estherchung/Desktop/Main_Data.csv"  # Update this with your dataset path
df = pd.read_csv(file_path)

# 🔹 Step 2: Define Features (X) and Target (y)
target_column = "RUL"  # Battery Remaining Useful Life
X = df.drop(columns=[target_column])
y = df[target_column]

# 🔹 Step 3: Split Data into Training and Testing (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 4: Define Hyperparameter Grids
rf_param_grid = {
    'n_estimators': [100, 200, 300],   # Number of trees in the forest
    'max_depth': [10, 20, None],       # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required in a leaf node
}

gb_param_grid = {
    'n_estimators': [100, 300, 500],   # Number of boosting stages
    'learning_rate': [0.01, 0.1, 0.2], # Shrinkage term
    'max_depth': [3, 5, 10]            # Maximum depth of individual estimators
}

# 🔹 Step 5: Tune Random Forest
print("Tuning Random Forest... ⏳")
rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Best Random Forest Model
best_rf = rf_grid_search.best_estimator_
print("\nBest Random Forest Parameters:", rf_grid_search.best_params_)

# 🔹 Step 6: Tune Gradient Boosting
print("\nTuning Gradient Boosting... ⏳")
gb_grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
gb_grid_search.fit(X_train, y_train)

# Best Gradient Boosting Model
best_gb = gb_grid_search.best_estimator_
print("\nBest Gradient Boosting Parameters:", gb_grid_search.best_params_)

# 🔹 Step 7: Evaluate the Best Models
models = {
    "Tuned Random Forest": best_rf,
    "Tuned Gradient Boosting": best_gb
}

results = []
for model_name, model in models.items():
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save results
    results.append({"Model": model_name, "MSE": mse, "MAE": mae, "R² Score": r2})

    print(f"\n{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")

# 🔹 Step 8: Save Best Models
os.makedirs("results/models/", exist_ok=True)
joblib.dump(best_rf, "results/models/best_random_forest.pkl")
joblib.dump(best_gb, "results/models/best_gradient_boosting.pkl")

print("\n✅ Best models saved in 'results/models/'")


# In[11]:


from sklearn.model_selection import GridSearchCV

# 🔹 Define Parameter Grid for Extra Trees
et_param_grid = {
    'n_estimators': [100, 200, 300],   # Number of trees in the forest
    'max_depth': [10, 20, None],       # Maximum depth of trees
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]      # Minimum samples in a leaf
}

# 🔹 Initialize GridSearchCV
print("Tuning Extra Trees Regressor... ⏳")
et_grid_search = GridSearchCV(ExtraTreesRegressor(random_state=42), et_param_grid, 
                              cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# 🔹 Fit GridSearchCV
et_grid_search.fit(X_train, y_train)

# 🔹 Get Best Extra Trees Model
best_et = et_grid_search.best_estimator_
print("\nBest Extra Trees Parameters:", et_grid_search.best_params_)

# 🔹 Evaluate the Best Model
y_pred = best_et.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTuned Extra Trees - MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")

# 🔹 Save the Best Model
import joblib
joblib.dump(best_et, "results/models/best_extra_trees.pkl")

print("\n✅ Best Extra Trees model saved in 'results/models/'")


# In[13]:


import pandas as pd
import joblib

# 🔹 Step 1: Load the trained model
model_path = "results/models/best_extra_trees.pkl"
best_et = joblib.load(model_path)

# 🔹 Step 2: Load new data (Predict_RUL.csv)
predict_file = "/Users/estherchung/Desktop/Predict_RUL.csv"  # Update path if needed
df_new = pd.read_csv(predict_file)

# 🔹 Step 4: Predict RUL using the trained model
predicted_rul = best_et.predict(df_new)

# Print first 5 predictions to verify output
print("First 5 Predicted RUL values:", predicted_rul[:5])

# 🔹 Step 5: Add Predictions to DataFrame and Save to CSV
df_new["Predicted RUL"] = predicted_rul
output_path = "/Users/estherchung/Desktop/predicted_RUL_results.csv"
df_new.to_csv(output_path, index=False)



# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Step 1: Load Dataset
file_path = "/Users/estherchung/Desktop/Main_Data.csv" 
df = pd.read_csv(file_path)

# 🔹 Step 2: Define Target Variable (RUL) and Features
features = ['Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)', 
            'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 
            'Charging time (s)']

# 🔹 Step 3: Create Scatter Plots (Each Feature vs. RUL)
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    plt.scatter(df[feature], df["RUL"], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel("RUL")
    plt.title(f"RUL vs {feature}")

plt.tight_layout()
plt.show()

# 🔹 Step 4: Compute Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df[features + ["RUL"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of RUL and Features")
plt.show()



# In[ ]:




