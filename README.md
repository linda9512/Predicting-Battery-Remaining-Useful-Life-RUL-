# Predicting Battery Remaining Useful Life (RUL) Using Machine Learning
Predicting the Remaining Useful Life (RUL) of batteries is crucial for ensuring reliability, safety, and efficiency in critical applications such as electric vehicles (EVs), aerospace, and energy storage systems. Unexpected battery failures can lead to operational disruptions and safety hazards, making accurate RUL prediction essential. In this study, we develop a machine learning-based approach to predict battery RUL using multiple regression models trained on a dataset containing features such as Discharge Time, Voltage Decrement, and Charging Time. We evaluate six models: Linear Regression, Random Forest, Extra Trees, Gradient Boosting, Neural Networks, and Support Vector Regression (SVR).
Found that Extra Trees model was performing the best-Using the tuned Extra Trees model, we predict RUL for 100 unseen battery samples and save the results. This study demonstrates the effectiveness of ensemble learning in battery RUL prediction, providing a robust tool for predictive maintenance and reliability assessment.


# 3 Different Data Uploaded
1️⃣ Main_Data.csv → The primary dataset used for training and testing the machine learning model.
2️⃣ Predict_RUL.csv → A dataset without RUL, where predictions need to be made using the trained model.
3️⃣ Predict_RUL_results.csv → The same as Predict_RUL.csv, but with an added Predicted RUL column from your model.
Battery RUL Code.py-> Python Code

# Instructions for data preparation
Step 1: Download Main_Data.csv from this GitHub
1.	Open the GitHub repository containing Main_Data.csv.
2.	Save it in a convenient location (e.g., Documents or Downloads).
   
Step 2: Preprocess the data using Jupyter Notebook
1.	Launch Jupyter Notebook on your laptop.
2.	Check for missing values in each column and remove any rows containing missing data. Then, identify and eliminate any duplicate rows in the dataset.
   
(Drops all rows with missing values)
df_cleaned = df.dropna()  

(Remove duplicate rows)
df_cleaned = df_cleaned.drop_duplicates()

Step 3: Save the Cleaned Data=>Save the cleaned dataset as  Main_Data.csv on your laptop.

# Python packages required to run the code
•	System and File Handling
o	os → For interacting with the operating system (file handling, paths)

•	Data Processing and Computation
o	pandas → For handling structured data (DataFrames, CSV files)
o	numpy → For numerical computations (arrays, mathematical operations)

•	Machine Learning and Model Handling
o	joblib → For saving and loading machine learning models efficiently
o	scikit-learn → For machine learning tasks, including: 
o	train_test_split, GridSearchCV → For data splitting and hyperparameter tuning
o	mean_squared_error, mean_absolute_error, r2_score → For model evaluation
o	LinearRegression → For linear regression models
o	RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor → For tree-based regression models
o	MLPRegressor → For neural network-based regression
o	SVR → For Support Vector Regression
o	StandardScaler → For feature scaling


•	Data Visualization
o	matplotlib → For creating basic plots and graphs
o	seaborn → For advanced statistical visualizations



#  Running Instructions

# Input Database
To run the analysis, ensure you have the following datasets stored in the appropriate location:
•	Main_Data.csv → The primary dataset used for training and testing the machine learning model.
•	Predict_RUL.csv → A dataset without RUL values, where predictions will be made using the trained model.

# Model Training & Evaluation
The dataset is split into training (80%) and testing (20%) sets to evaluate model performance. We train and test the following six machine learning models:
•	Linear Regression
•	Random Forest
•	Extra Trees
•	Gradient Boosting
•	Neural Network (MLP Regressor)
•	Support Vector Regression (SVR)
Each model is assessed using standard regression metrics:
•	Mean Squared Error (MSE)
•	Mean Absolute Error (MAE)
•	R² Score
Among these models, Extra Trees demonstrated the best predictive performance and was selected for further fine-tuning.

# Model Fine-Tuning
To optimize performance, we fine-tune the Random Forest, Gradient Boosting, and Extra Trees models(Because these 3 models had the highest performance values) by adjusting:
•	Number of estimators
•	Depth of trees
•	Learning rate
•	Other relevant hyperparameters
After tuning, Extra Trees continued to outperform the other models, proving its robustness in predicting battery RUL.

# Make Predictions
Once the model is trained, generate predictions for the unseen dataset (Predict_RUL.csv). This dataset contains all the features present in the main dataset, except for the RUL column, as our goal is to predict RUL.
This will output a file:
Predict_RUL_results.csv → Contains the predicted RUL values.

# Feature Importance Analysis
To interpret the relationships between RUL and battery characteristics, we perform:
•	Correlation Matrices → Identify key factors influencing battery lifespan.
•	Scatter Plots → Visualize feature relationships with RUL.














