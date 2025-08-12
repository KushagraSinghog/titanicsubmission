                                               🚢 Titanic Survival Prediction

A Machine Learning Pipeline using XGBoost, LightGBM & Stacking Classifiers

                                                📌 Project Overview

This project predicts whether a passenger survived the Titanic disaster using machine learning.
It follows the Kaggle Titanic: Machine Learning from Disaster challenge format.

We train multiple models, tune hyperparameters, and combine them using stacking for improved performance.


                                                📂 Dataset Description

The project uses two CSV files:

train.csv → Passenger data with the target variable Survived.

test.csv → Passenger data without the Survived column (predictions needed).


                                                 🛠 Workflow

                 1️⃣ Data Loading & Exploration

Load datasets with pandas.

Display sample rows & dataset info.

Analyze numerical feature correlations with survival rate.

                 2️⃣ Data Preprocessing

Numerical Features:

    Missing values filled with median (SimpleImputer).

Categorical Features:

    Missing Embarked values replaced with the most frequent value (mode).

Categorical columns (Sex, Embarked) encoded with OrdinalEncoder.

Dropped unused columns: Name, Ticket, Cabin.

Combined processed numerical and categorical data into final training and test sets.


                  3️⃣ Models Used

1. XGBoost (XGBClassifier)

2. LightGBM (LGBMClassifier)

3. Stacking Classifier

Base models: LightGBM & XGBoost

Final estimator: Logistic Regression

                   4️⃣ Hyperparameter Tuning

Used GridSearchCV to tune:

1. n_estimators

2. max_depth

Evaluated models using 5-fold cross-validation accuracy.


                   5️⃣ Model Evaluation

Accuracy scores printed for:

1. LightGBM

2. XGBoost

3. Stacked Model

                   6️⃣ Test Set Predictions

Applied same preprocessing steps to the test set.

Generated predictions using the stacked model.

Predictions stored in test_pred
