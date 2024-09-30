Sensor Fault Detection Project
This project focuses on detecting faults in sensors using machine learning techniques. It leverages a dataset containing sensor failure data to train a model for accurate fault prediction. The project employs preprocessing techniques, oversampling, and different machine learning models.

Project Overview
The purpose of this project is to identify faults in sensor data using machine learning. The model is trained on a dataset of sensor failures and classifies the data into two categories: "pos" (fault detected) and "neg" (no fault detected).

Dataset
The dataset for this project is large and can be accessed via the following link:

Download the Dataset

Problem Statement
The system in focus is the Air Pressure System (APS) in trucks. The goal is to minimize false predictions that lead to unnecessary repairs or missed failures. The cost of missing a faulty truck (Cost 2 = 500) is much higher than the cost of unnecessary checks (Cost 1 = 10). Therefore, the model aims to reduce false negatives significantly.

The total cost can be computed as:

mathematica
Copy code
Total Cost = Cost_1 * Number of False Positives + Cost_2 * Number of False Negatives
Challenges
Handling numerous missing values in the dataset.
Managing the class imbalance in the target variable.
Reducing false negatives is critical due to the high cost of missed failures.
Project Files
pipeline_training.ipynb: This notebook covers the preprocessing and training pipeline, including handling missing data, scaling, and oversampling using SMOTE.

Scania_APS_failure_prediction.ipynb: This notebook provides an in-depth exploration of various imputation strategies and model performance evaluations.

Key Experiments:
Experiment 1: KNN Imputer

Uses the KNN imputer to handle missing values. The best model for this experiment was the XGBoost classifier with the lowest total cost of 4460.
Experiment 2: Simple Imputer (Median Strategy)

Replaces missing values with the median. CatBoost was the best model for this experiment.
Experiment 3: MICE Imputation

Uses the MICE algorithm to handle missing values. XGBoost gave the best results for this approach, with a total cost of 3510.
Experiment 4: Simple Imputer (Constant Strategy)

Replaces missing values with a constant value (0). XGBoost performed best, with the lowest total cost of 2950.
Experiment 5: Simple Imputer (Mean Strategy)

Missing values are replaced with the mean. XGBoost was the best model with a cost of 4950.
Experiment 6: Principal Component Analysis (PCA)

PCA was used to reduce the dimensionality of the dataset. Random Forest performed best in this setup, though the cost was significantly higher at 34150.
Final Model:
The XGBoost Classifier with Simple Imputer (constant strategy) performed the best, achieving 99.6% accuracy and a total cost of 2950.

Evaluation Metrics:
Accuracy
F1-Score
Precision
Recall
ROC-AUC
Total cost based on false positives and false negatives.
trained_model.joblib: This file contains the saved XGBoost model, which can be loaded and used for making predictions on new data.

untitled9.py: This Python file contains the code for the Streamlit app that allows users to upload sensor data and predict whether a truck requires service based on the pre-trained XGBoost model.

Key Features:
User Input: Allows users to upload a CSV file of sensor data.
Preprocessing: The uploaded data is preprocessed by replacing 'na' values with 0 and converting columns to numeric format.
Model Prediction: The app uses the pre-trained model to predict whether the truck requires servicing.
Report Generation: Generates a report listing all trucks and corresponding remarks, indicating whether a truck is fine or needs service.
Running the App:
To run the app, you need to install Streamlit:

bash
Copy code
pip install streamlit
Then, you can run the app using the command:

bash
Copy code
streamlit run untitled9.py
