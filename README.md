# Predicting-Battery-Remaining-Useful-Life-RUL-
Predicting the Remaining Useful Life (RUL) of batteries is crucial for ensuring reliability, safety, and efficiency in critical applications such as electric vehicles (EVs), aerospace, and energy storage systems. Unexpected battery failures can lead to operational disruptions and safety hazards, making accurate RUL prediction essential. In this study, we develop a machine learning-based approach to predict battery RUL using multiple regression models trained on a dataset containing features such as Discharge Time, Voltage Decrement, and Charging Time. We evaluate six models: Linear Regression, Random Forest, Extra Trees, Gradient Boosting, Neural Networks, and Support Vector Regression (SVR).
Found that Extra Trees model was performing the best-Using the tuned Extra Trees model, we predict RUL for 100 unseen battery samples and save the results. This study demonstrates the effectiveness of ensemble learning in battery RUL prediction, providing a robust tool for predictive maintenance and reliability assessment.


3 Different Data Uploaded
1️⃣ Main_Data.csv → The primary dataset used for training and testing the machine learning model.
2️⃣ Predict_RUL.csv → A dataset without RUL, where predictions need to be made using the trained model.
3️⃣ Predict_RUL_results.csv → The same as Predict_RUL.csv, but with an added Predicted RUL column from your model.
