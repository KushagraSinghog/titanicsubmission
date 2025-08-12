import pandas as pd 
import numpy as np 
import matplotlib_inline 
import matplotlib.pyplot as plt 

test_set=pd.read_csv('test.csv') 
train_set=pd.read_csv('train.csv') 

print(test_set.head())
print(test_set.info())  

train_corr=train_set.select_dtypes(include=[np.number]) 
corr_matrix=train_corr.corr() 
print(corr_matrix["Survived"].sort_values(ascending=False)) 

train_set_labels=train_set["Survived"]

from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(strategy="median") 
train_num=train_set.drop(["Survived", "Sex", "Embarked", "Ticket", "Name", "Cabin"], axis=1) 
imputer.fit(train_num)  
print(imputer.statistics_)
x=imputer.transform(train_num)  
train_num_tr=pd.DataFrame(x, columns=train_num.columns, index=train_num.index) 
print(train_num_tr) 

from sklearn.preprocessing import OrdinalEncoder 
oe=OrdinalEncoder() 
train_set["Embarked"].fillna(train_set["Embarked"].mode()[0], inplace=True)
train_cat=train_set[["Sex", "Embarked"]] 
train_cat_en=oe.fit_transform(train_cat) 
print(train_cat_en[:40]) 
train_cat_tr=pd.DataFrame(train_cat_en, columns=["Sex", "Embarked"], index=train_set.index) 
print(train_cat_tr) 

train_set_prepared=pd.concat([train_num_tr, train_cat_tr], axis=1) 
print(train_set_prepared.head(10)) 

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
xgbc = XGBClassifier(n_estimators=210, learning_rate=0.01, max_depth=9, subsample=0.6, colsample_bytree=0.8, random_state=42) 
xgbc_pred = cross_val_predict(xgbc, train_set_prepared, train_set_labels, cv=5)

from sklearn.model_selection import GridSearchCV 
param_grid={ 'n_estimators': [120, 125, 130], 'max_depth': [4, 5, 6] } 
search=GridSearchCV(estimator=xgbc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1) 
search.fit(train_set_prepared, train_set_labels) 
print("best parameters: ", search.best_params_) 
print("best score: ", search.best_score_)  

from lightgbm import LGBMClassifier 
lgbm=LGBMClassifier(n_estimators=100, learning_rate=0.01, max_depth=5, random_state=42, force_row_wise = True) 
lgbm.fit(train_set_prepared, train_set_labels) 
lgbm_pred=cross_val_predict(lgbm, train_set_prepared, train_set_labels, cv=5) 
print("predictions= ", lgbm_pred[:20]) 
print("labels= ", list(train_set_labels[:20]))  

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
stack = StackingClassifier(estimators=[('lgbm', lgbm), ('xgb', xgbc) ], final_estimator=LogisticRegression() ) 
stack.fit(train_set_prepared, train_set_labels)
stack_pred = cross_val_predict(stack, train_set_prepared, train_set_labels, cv=5)

from sklearn.metrics import accuracy_score 
acc=accuracy_score(train_set_labels, lgbm_pred)  
print(acc) 
acc3 = accuracy_score(train_set_labels, xgbc_pred) 
print(acc3)  
acc5 = accuracy_score(train_set_labels, stack_pred) 
print(acc5) 

test_corr=test_set.select_dtypes(include=[np.number]) 
corr2_matrix=test_corr.corr() 
 

test_num=test_set.drop(["Sex", "Embarked", "Ticket", "Name", "Cabin"], axis=1) 
print(imputer.statistics_)
y=imputer.transform(test_num)  
test_num_tr=pd.DataFrame(y, columns=test_num.columns, index=test_num.index) 
print(test_num_tr) 

test_set["Embarked"].fillna(test_set["Embarked"].mode()[0], inplace=True)
test_cat=test_set[["Sex", "Embarked"]] 
test_cat_en=oe.transform(test_cat) 
print(test_cat_en[:40]) 
test_cat_tr=pd.DataFrame(test_cat_en, columns=["Sex", "Embarked"], index=test_set.index) 
print(test_cat_tr) 

test_set_prepared=pd.concat([test_num_tr, test_cat_tr], axis=1) 
print(test_set_prepared.head(10)) 

test_pred=stack.predict(test_set_prepared)  

