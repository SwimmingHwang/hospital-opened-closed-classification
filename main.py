"""
Process
1. file read
2. data preprocessing
3. Generate models and tune the model's hyperparameters
    - xgboost
    - lightgbm
    - RandomForest
4. ensemble modeling
5. Show accuracy
"""

'''
# conda install -c conda-forge shap
# pip install xgboost
# pip install lightgbm
'''
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import shap
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report

'''
Read files and data preprocessing
- train 데이터랑 test 데이터 합한 후 전처리 동시에 진행 
- output 컬럼인 'OC'가 비어있으면(==0) test 데이터 임을 활용하여 후에 다시 구분
'''
# Reading the train and test files
train_prod_df = pd.read_csv('data\\train.csv')
test_prod_df = pd.read_csv('data\\test_empty.csv')
# Removing the comma in the employee1 and 2 columns in the test dataset and replace it with empty space and convert it to float format.
test_prod_df.employee1 = test_prod_df.employee1.astype('str').str.replace(",", "").astype('float')
test_prod_df.employee2 = test_prod_df.employee2.astype('str').str.replace(",", "").astype('float')

# Converting the employee1 and 2 column as float in the train set as done for the test dataset
train_prod_df.employee1 = train_prod_df.employee1.astype('float')
train_prod_df.employee2 = train_prod_df.employee2.astype('float')
train_prod_df.OC = train_prod_df.OC.astype('str').str.replace(" ", "")

# Combining the train and test dataset
train_test_prod = train_prod_df.append(test_prod_df)

# Get the object and numeric columns seperately
factor_columns = train_test_prod.select_dtypes(include=['object']).columns
numeric_columns = train_test_prod.columns.difference(factor_columns)

# After analysis realized that the bed counts of these two hospitals may have had wrong entries.
# Filling up the empty instkind and bedCount for hospital id 430 and 413
train_test_prod.loc[train_test_prod.inst_id == 430, ['instkind']] = 'dental_clinic'
train_test_prod.loc[train_test_prod.inst_id == 430, ['bedCount']] = 0
train_test_prod.loc[train_test_prod.inst_id == 413, ['bedCount']] = -999

# Fill the empty values in the object columns as "Not sure"
train_test_prod[factor_columns] = train_test_prod[factor_columns].fillna('Not_sure')
# Fill all the empty values in the numeric columns as -999
train_test_prod[numeric_columns] = train_test_prod[numeric_columns].fillna(-999)

# Convert all the object columns to numeric since the ML algorithms don't accept object features directly
fac_le = LabelEncoder()
train_test_prod[factor_columns] = train_test_prod.loc[:, factor_columns].apply(lambda x: fac_le.fit_transform(x))

# Splitting back data to train prod and test prod
train_prod = train_test_prod.loc[train_test_prod.OC != 0,]
test_prod = train_test_prod.loc[train_test_prod.OC == 0,]
train_prod['OC'] = train_prod['OC'] - 1

# Obtain the submission ID to create the submission file later
sub_id = test_prod.inst_id

# Get the dependent and independent column
dep = 'OC'
indep = train_prod.columns.difference([dep])

train_prod_X = train_prod[indep]
train_prod_Y = train_prod[dep]
test_prod_X = test_prod[indep]

'''
Hyperparameter Tuning the Random Forest
'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=1, stop=25, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the random search model
rf_random.fit(train_prod_X, train_prod_Y)

############################################################################
############ Random Forest with hyper-parameter tuning
############################################################################
rf_estimators_tune = rf_random.best_estimator_.n_estimators
rf_max_depth_tune = rf_random.best_estimator_.max_depth
rf_max_features_tune = rf_random.best_estimator_.max_features

np.random.seed(100)
RF_prod_tune = RandomForestClassifier(n_estimators=rf_estimators_tune,
                                 max_depth=rf_max_depth_tune,
                                 max_features=rf_max_features_tune)
RF_prod_model_tune = RF_prod_tune.fit(train_prod_X, train_prod_Y)
RF_prod_prediction_tune = RF_prod_tune.predict_proba(test_prod_X)[:, 1]

sub_RF_tune = pd.DataFrame({'inst_id': sub_id, 'OC': RF_prod_prediction_tune})
sub_RF_tune = sub_RF_tune[['inst_id', 'OC']]

sub_RF_tune['OC'] = [1 if oc >= 0.7 else 0 for oc in sub_RF_tune['OC']]

############################################################################
############ Random Forest (튜닝 전)
############################################################################
rf_estimators = 10
np.random.seed(100)
RF_prod = RandomForestClassifier(n_estimators=rf_estimators)
RF_prod_model = RF_prod.fit(train_prod_X, train_prod_Y)
RF_prod_prediction = RF_prod.predict_proba(test_prod_X)[:, 1]

sub_RF = pd.DataFrame({'inst_id': sub_id, 'OC': RF_prod_prediction})
sub_RF = sub_RF[['inst_id', 'OC']]
sub_RF['OC'] = [1 if oc >= 0.7 else 0 for oc in sub_RF['OC']]

# 튜닝 전 후 classification 성능 비교
print("-" * 50)
print("튜닝 전 후 classification 성능 비교")

test_prod_tmp = test_prod[['inst_id', 'OC']]  # 테스트 데이터 OC value 세팅
close_idx = [5, 6, 24, 30, 64, 123, 229, 258, 293, 341, 425, 429, 431]
test_prod['OC'] = [0 if id in close_idx else 1 for id in test_prod['inst_id']]
