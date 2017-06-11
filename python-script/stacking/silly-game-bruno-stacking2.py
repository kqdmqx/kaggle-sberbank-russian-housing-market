# coding=utf-8

import numpy as np
import pandas as pd
# import xgboost as xgb
import sys
sys.path.append('../..')
from my_py_models.stacking import Stacking
from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from os.path import join
from sklearn.metrics import mean_squared_error

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv(join(INPUT_PATH, 'train.csv'), parse_dates=['timestamp'])
df_test = pd.read_csv(join(INPUT_PATH, 'test.csv'), parse_dates=['timestamp'])
train_id = pd.read_csv(join(OUTPUT_PATH, 'sample/TrainId-SillyGame-Gunja-Clean.csv'))
df_train = df_train.loc[df_train.id.isin(train_id.id.unique())].copy()

df_macro = pd.read_csv(join(INPUT_PATH, 'macro.csv'), parse_dates=['timestamp'])
df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

mult = .969
y_train = df_train['price_doc'].values * mult + 10
train_id = df_train.id
test_id = df_test.id
df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)

# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

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
pred_oof, pred_test = stacking.fit_predict(X_train, y_train, X_test)

for pred_oof_single in pred_oof.T:
    print np.sqrt(mean_squared_error(np.log(pred_oof_single + 1), np.log(y_train + 1)))

df_sub = pd.DataFrame({'id': test_id, 'price_doc': pred_test[:, 0]})
df_sub.to_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-Bruno-Stacking2-2017061100-Test.csv'), index=False)

df_oof = pd.DataFrame({'id': train_id, 'price_doc': pred_oof[:, 0]})
df_oof.to_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-Bruno-Stacking2-2017061100-OutOfFold.csv'), index=False)
