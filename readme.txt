#                                                                     Titanic Survival Prediction - Machine Learning Project

This project predicts passenger survival on the Titanic using machine learning. It is based on the Kaggle Titanic Competition(https://www.kaggle.com/competitions/titanic).


##                                         Dataset Description

   The project uses three datasets:

- 'train.csv' — training data containing features and survival outcomes.
- 'test.csv' — test data with missing `Survived` values to predict.
- 'gender_submission.csv' — sample submission (not directly used in model training).

---

##                                         Project Workflow

### 1. Data Loading
- The train, test, and gender submission CSV files are loaded using pandas.

### 2. Data Exploratory Analysis
- Dataset inspection with '.head()', '.info()', '.hist()', and '.isnull().sum()' 

### 3. Preprocessing
- Numerical columns ('Age', 'Fare', etc.) are imputed with the median.
- Categorical columns ('Sex', 'Embarked') are imputed with the most frequent value and encoded using 'OrdinalEncoder'.
- Textual columns like 'Name', 'Ticket', and 'Cabi'` are dropped as they're not used in the model.

### 4. Training Model
- Model: 'LightGBMClassifier' from LightGBM.
- Parameters: 'n_estimators=100', 'learning_rate=0.01', 'max_depth=5', 'random_state=42' .
- Training is done on the preprocessed training set.
- Evaluation is done using 'cross_val_predict' and 'accuracy_score' methods.

### 5. Test Set Prediction
- The same preprocessing steps are applied to the test set.
- Predictions are made using the trained LightGBM model.

### 6. Output
- A CSV file 'titanic_final_submission.csv' is generated for submission.
- This file includes:
  - 'PassengerId' 
  - 'Survived' prediction (0 = Not Survived, 1 = Survived)

---

## Requirements

Install the following Python packages (via pip or conda):

```bash
pip install pandas numpy matplotlib scikit-learn lightgbm 
