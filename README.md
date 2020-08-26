# hospital-opened-closed-classification
Machine Learning Exercise(2020.08)

## 1. 주제 및 목표
병원 폐업 여부를 예측하여 대출 승인여부 결정
## 2. 배경
- 한국 핀테크 기업 모우다(MOUDA): 상환기간 동안의 계속 경영 여부를 예측하여 병원들에게 금융 기회를 제공   
- 일반적으로 병원 대출 시 신용점수 또는 담보물을 위주로 평가를 진행했던 기존 금융기관과의 차별점   
- 신용 점수가 낮거나 담보를 가지지 못하는 우수 병원들에게도 금융 기회를 제공하자는 취지   
## 3. 활용 데이터
- 의료기관의 폐업 여부가 포함된 최근 2개년의 재무정보와 병원 기본정보   
- (출처) Dacon 병원 개/폐업 분류 예측 경진대회 (https://dacon.io/competitions/official/9565/data/)
## 4. 개발 환경
1. Environment
anaconda3 + python 3.8
2. library
```
conda install -c conda-forge shap
pip install xgboost
pip install lightgbm
```

```
import os
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
```
