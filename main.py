import pandas as pd                                           #importing necessary classes
import numpy as np 
import matplotlib_inline 
import matplotlib.pyplot as plt 

test_set=pd.read_csv('test.csv')                              #reading the files
train_set=pd.read_csv('train.csv') 
gender=pd.read_csv('gender_submission.csv') 

#print(test_set.head())                                       #exploring the data
#print(test_set.info())  
#print(train_set.head(40)) 
#print(train_set.info()) 
#print(gender.head(40)) 
#print(gender.info()) 
#train_set.hist() 
#plt.show()  
#test_set.hist() 
#plt.show() 
#gender.hist() 
#plt.show() 
#print(train_set.isnull().sum()) 
#print(test_set.isnull().sum()) 

train_corr=train_set.select_dtypes(include=[np.number])                   #creating a correlation matrix for training set
corr_matrix=train_corr.corr() 
#print(corr_matrix["Survived"].sort_values(ascending=False)) 

train_set_labels=train_set["Survived"]                                    #seperating the labels

from sklearn.impute import SimpleImputer                                  #filling the missing numerical values on training set
imputer=SimpleImputer(strategy="median") 
train_num=train_set.drop(["Survived", "Sex", "Embarked", "Ticket", "Name", "Cabin"], axis=1)                  #dropping text and categorical columns
imputer.fit(train_num)  
#print(imputer.statistics_)
x=imputer.transform(train_num)  
train_num_tr=pd.DataFrame(x, columns=train_num.columns, index=train_num.index)                                #converting the result to dataframe
#print(train_num_tr) 

from sklearn.preprocessing import OrdinalEncoder                          #encoding the text attributes in training set
oe=OrdinalEncoder() 
train_set["Embarked"].fillna(train_set["Embarked"].mode()[0], inplace=True)                                   #filling the embarked column in training set
train_cat=train_set[["Sex", "Embarked"]] 
train_cat_en=oe.fit_transform(train_cat) 
#print(train_cat_en[:40]) 
train_cat_tr=pd.DataFrame(train_cat_en, columns=["Sex", "Embarked"], index=train_set.index)                   #converting the result to dataframe
#print(train_cat_tr) 

train_set_prepared=pd.concat([train_num_tr, train_cat_tr], axis=1)                                            #training set prepared
#print(train_set_prepared.head(10)) 

from sklearn.model_selection import cross_val_predict                                                         #using lgbm classifier model for prediction on training set
from lightgbm import LGBMClassifier 
lgbm=LGBMClassifier(n_estimators=100, learning_rate=0.01, max_depth=5, random_state=42) 
lgbm.fit(train_set_prepared, train_set_labels) 
lgbm_pred=cross_val_predict(lgbm, train_set_prepared, train_set_labels, cv=5) 
#print("predictions= ", lgbm_pred[:20]) 
#print("labels= ", list(train_set_labels[:20]))  

from sklearn.metrics import accuracy_score                                                                    #for checking accuracy
acc=accuracy_score(train_set_labels, lgbm_pred) 
print(acc)       



test_corr=test_set.select_dtypes(include=[np.number])                     #creating correlation matrix for test set
corr2_matrix=test_corr.corr() 
#print(corr2_matrix["Survived"].sort_values(ascending=False)) 

test_num=test_set.drop(["Sex", "Embarked", "Ticket", "Name", "Cabin"], axis=1)                                #filling the missing numerical values on training set while dropping text and categorical columns  
#print(imputer.statistics_)
y=imputer.transform(test_num)  
test_num_tr=pd.DataFrame(y, columns=test_num.columns, index=test_num.index) 
#print(test_num_tr) 
  
test_set["Embarked"].fillna(test_set["Embarked"].mode()[0], inplace=True)                                     #filling the embarked column in test set and encoding the text attributes
test_cat=test_set[["Sex", "Embarked"]] 
test_cat_en=oe.transform(test_cat)                                                                            
#print(test_cat_en[:40]) 
test_cat_tr=pd.DataFrame(test_cat_en, columns=["Sex", "Embarked"], index=test_set.index)                      #converting the result to dataframe
#print(train_cat_tr) 

test_set_prepared=pd.concat([test_num_tr, test_cat_tr], axis=1)                                               #test set prepared
#print(test_set_prepared.head(10))      

test_pred=lgbm.predict(test_set_prepared)                                  #applying the same model on test set for predictions

final=pd.DataFrame( {"PassengerId": test_set["PassengerId"], "Survived": test_pred} )                         #converting the final predictions to a csv file
final.to_csv("titanic_final_submission.csv", index=False)  

