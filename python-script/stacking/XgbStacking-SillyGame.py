# coding=utf-8

import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from my_py_models.stacking import Stacking
from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from os.path import join
from sklearn.metrics import mean_squared_error

# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import model_selection, preprocessing
# import xgboost as xgb
# import datetime

louis_oof = pd.read_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-Louis-Stacking2-2017061100-OutOfFold.csv'), index_col='id')
bruno_oof = pd.read_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-Bruno-Stacking2-2017061100-OutOfFold.csv'), index_col='id')
gunja_oof = pd.read_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyDataBaseLine-GunJa-Stacking2-20170606900-OutOfFold-ToBeFixed.csv'), index_col='id')

louis_test = pd.read_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-Louis-Stacking2-2017061100-Test.csv'), index_col='id')
bruno_test = pd.read_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-Bruno-Stacking2-2017061100-Test.csv'), index_col='id')
gunja_test = pd.read_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyDataBaseLine-GunJa-Stacking2-2017060900-Test.csv'), index_col='id')

train = pd.read_csv(join(INPUT_PATH, 'train.csv'))
test = pd.read_csv(join(INPUT_PATH, 'test.csv'))

train_id = pd.read_csv(join(OUTPUT_PATH, 'sample/TrainId-SillyGame-Gunja-Clean.csv'))
train = train.loc[train.id.isin(train_id.id.unique())].copy()


train_stack = pd.DataFrame(
    {
        'louis': louis_oof.price_doc,
        'bruno': bruno_oof.price_doc,
        'gunja': gunja_oof.price_doc,
        'price_doc': train.set_index('id').price_doc,
    }
)

test_stack = pd.DataFrame(
    {
        'louis': louis_test.price_doc,
        'bruno': bruno_test.price_doc,
        'gunja': gunja_test.price_doc
    }
)

train_id = train_stack.reset_index().id
test_id = test_stack.reset_index().id


mult = .969
y_train = train_stack.price_doc.values * mult + 10
X_train = train_stack.drop('price_doc', axis=1).values
X_test = test_stack.values

xgb_params = {
    'eta': 0.02,
    'max_depth': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


clf = MyXgbClassifier2(xgb_params)
stacking = Stacking(5, [clf])
pred_oof, pred_test = stacking.fit_predict(X_train, y_train, X_test)


for pred_oof_single in pred_oof.T:
    print np.sqrt(mean_squared_error(np.log(pred_oof_single), np.log(y_train)))

df_sub = pd.DataFrame({'id': test_id, 'price_doc': pred_test[:, 0]})
df_sub.to_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-XgbStackingTrans-2017061200-Test.csv'), index=False)

df_oof = pd.DataFrame({'id': train_id, 'price_doc': pred_oof[:, 0]})
df_oof.to_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-XgbStackingTrans-2017061200-OutOfFold.csv'), index=False)
