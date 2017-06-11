# coding=utf-8

# import matplotlib.pyplot as plt
# import seaborn as sns
# import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
import sys
sys.path.append('../..')
from my_py_models.stacking import Stacking
from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from os.path import join
from sklearn.metrics import mean_squared_error


train = pd.read_csv(join(INPUT_PATH, 'train.csv'))
test = pd.read_csv(join(INPUT_PATH, 'test.csv'))

train_id = pd.read_csv(join(OUTPUT_PATH, 'sample/TrainId-SillyGame-Gunja-Clean.csv'))
train = train.loc[train.id.isin(train_id.id.unique())].copy()

train_id = train.id
test_id = test.id

mult = .969
y_train = train["price_doc"] * mult + 10
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


clf = MyXgbClassifier2(xgb_params)
stacking = Stacking(5, [clf])
pred_oof, pred_test = stacking.fit_predict(x_train, y_train, x_test)

for pred_oof_single in pred_oof.T:
    print np.sqrt(mean_squared_error(np.log(pred_oof_single + 1), np.log(y_train + 1)))

df_sub = pd.DataFrame({'id': test_id, 'price_doc': pred_test[:, 0]})
df_sub.to_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-Louis-Stacking2-2017061100-Test.csv'), index=False)

df_oof = pd.DataFrame({'id': train_id, 'price_doc': pred_oof[:, 0]})
df_oof.to_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-Louis-Stacking2-2017061100-OutOfFold.csv'), index=False)
