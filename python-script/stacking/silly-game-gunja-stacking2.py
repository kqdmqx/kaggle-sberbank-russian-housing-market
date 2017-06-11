# coding: utf-8

import pandas as pd
import numpy as np
import xgboost as xgb
import sys
sys.path.append('../..')
from my_py_models.stacking import Stacking
from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from sklearn.metrics import mean_squared_error

# load data
train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

train_id = train.id
test_id = test.id
train_columns = train.columns.tolist()
test_columns = test.columns.tolist()


def tridx(col):
    if col not in train_columns:
        return -1
    return train_columns.index(col)


def tsidx(col):
    if col not in test_columns:
        return -1
    return test_columns.index(col)


# 纠正个别错误
# @life_sq
equal_index = [601, 1896, 2791]
test.iloc[equal_index, tsidx('life_sq')] = test.iloc[
    equal_index, tsidx('full_sq')]

# @build_year
kitch_is_build_year = [13117]
train.iloc[kitch_is_build_year, tridx('build_year')] = train.iloc[
    kitch_is_build_year, tridx('kitch_sq')]

# @state
train.loc[train.state == 33, 'state'] = np.NaN

# 把离群点替换为np.NaN
train_test = pd.concat([train, test])
is_train = train_test.id.isin(train_id.unique())
is_test = train_test.id.isin(test_id.unique())

# @life_sq
isbad_life_sq = ((train_test.life_sq > train_test.full_sq) |
                 (train_test.life_sq < 5) |
                 ((train_test.life_sq > 300) & is_train) |
                 ((train_test.life_sq > 200) & is_test))
isbad_life_sq_id = train_test.loc[isbad_life_sq, 'id']
train_test.loc[isbad_life_sq, 'life_sq'] = np.NaN
print 'bad_life_sq', np.sum(isbad_life_sq)

# @full_sq
isbad_full_sq = ((train_test.full_sq < 5) |
                 ((train_test.full_sq > 210) & (train_test.life_sq / train_test.full_sq < 0.3) & is_train) |
                 ((train_test.full_sq > 150) & (train_test.life_sq / train_test.full_sq < 0.3) & is_test) |
                 ((train_test.life_sq > 300) & is_train) |
                 ((train_test.life_sq > 200) & is_test))
isbad_full_sq_id = train_test.loc[isbad_full_sq, 'id']
train_test.loc[isbad_full_sq, 'full_sq'] = np.NaN
print 'bad_full_sq', np.sum(isbad_full_sq)

# @kitch_sq
isbad_kitch_sq = ((train_test.id == 13120) |
                  (train_test.kitch_sq > train_test.life_sq) |
                  (train_test.kitch_sq == 0) |
                  (train_test.kitch_sq == 1))
isbad_kitch_sq_id = train_test.loc[isbad_kitch_sq, 'id']
train_test.loc[isbad_kitch_sq, 'kitch_sq'] = np.NaN
print 'bad_kitch_sq', np.sum(isbad_kitch_sq)

# @build_year
isbad_build_year = ((train_test.build_year < 1500) |
                    (train_test.build_year > 2200))
isbad_build_year_id = train_test.loc[isbad_build_year, 'id']
train_test.loc[isbad_build_year, 'build_year'] = np.NaN
print 'bad_build_year', np.sum(isbad_build_year)

# @num_room
isbad_num_room_selected_id = train_test.iloc[
    [10076, 11621, 17764, 19390, 24007, 26713, 29172, 3174, 7313]].id.unique()
isbad_num_room = ((train_test.id.isin(isbad_num_room_selected_id)) |
                  (train_test.num_room == 0))
isbad_num_room_id = train_test.loc[isbad_num_room, 'id']
train_test.loc[isbad_num_room, 'num_room'] = np.NaN
print 'bad_num_room', np.sum(isbad_num_room)

# @floor
isbad_floor = ((train_test.floor > train_test.max_floor) |
               (train_test.floor == 0))
isbad_floor_id = train_test.loc[isbad_floor, 'id']
train_test.loc[isbad_floor, 'num_room'] = np.NaN
print 'bad_floor', np.sum(isbad_floor)

# @max_floor
isbad_max_floor = ((train_test.floor > train_test.max_floor) |
                   (train_test.max_floor == 0))
isbad_max_floor_id = train_test.loc[isbad_max_floor, 'id']
train_test.loc[isbad_max_floor, 'num_room'] = np.NaN
print 'max_floor', np.sum(isbad_max_floor)


# 增加特征
# Add month-year
train_test['month_year_cnt'] = train_test.timestamp.dt.year * \
    100 + train_test.timestamp.dt.month

# Add week-year count
train_test['week_year_cnt'] = train_test.timestamp.dt.year * \
    100 + train_test.timestamp.dt.weekofyear

# Add month and day-of-week
train_test['month'] = train_test.timestamp.dt.month
train_test['dow'] = train_test.timestamp.dt.dayofweek

# Other feature engineering
train_test['rel_floor'] = train_test['floor'] / \
    train_test['max_floor'].astype(float)
train_test['rel_kitch_sq'] = train_test[
    'kitch_sq'] / train_test['full_sq'].astype(float)

train_test.apartment_name = train_test.sub_area + \
    train_test['metro_km_avto'].astype(str)
train_test['room_size'] = train_test[
    'life_sq'] / train_test['num_room'].astype(float)

# Deal with categorical values
train_test_numeric = train_test.select_dtypes(exclude=['object'])
train_test_obj = train_test.select_dtypes(include=['object']).copy()

for c in train_test_obj:
    train_test_obj[c] = pd.factorize(train_test_obj[c])[0]

train_test_values = pd.concat([train_test_numeric, train_test_obj], axis=1)


# 从训练集中去除预测目标离群的样本
train = train_test_values.loc[train_test_values.id.isin(train_id.unique())]
test = train_test_values.loc[train_test_values.id.isin(test_id.unique())]

train_outlier = ((train.price_doc / train.full_sq > 600000) |
                 (train.price_doc / train.full_sq < 10000))
train_outlier_id = train.loc[train_outlier].id
print 'train_outlier', np.sum(train_outlier)

train_clean = train.loc[~train_outlier].copy()
train_clean_id = train_clean['id']
test = test['id']
train_ex = train.loc[train_outlier].copy()


# # 来自sillyGame
# y_train = train_clean["price_doc"]
# x_train = train_clean.drop(["id", "timestamp", "price_doc"], axis=1)
# x_test = test.drop(["id", "timestamp", "price_doc"], axis=1)

# xgb_params = {
#     'eta': 0.05,
#     'max_depth': 5,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse',
#     'silent': 0,
#     'booster': 'gbtree',
#     'tuneLength': 3
# }

# dtrain = xgb.DMatrix(x_train, y_train)
# dtest = xgb.DMatrix(x_test)
# model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=300)
# y_predict = model.predict(dtest)
# y_predict = np.round(y_predict * 0.99)
# gunja_output = pd.DataFrame({'id': test_id, 'price_doc': y_predict})
# gunja_output.to_csv(
#     '../output/Submission-SillyDataBaseLine-GunJa.csv', index=False)


# 用5折CV生成折外预测
y_train = train_clean["price_doc"]
x_train = train_clean.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0,
    'booster': 'gbtree',
    'tuneLength': 3
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

clf = MyXgbClassifier2(xgb_params)
stacking = Stacking(5, [clf])
pred_oof, pred_test = stacking.fit_predict(x_train, y_train, x_test)


for pred_oof_single in pred_oof.T:
    print np.sqrt(mean_squared_error(np.log(pred_oof_single + 1), np.log(y_train + 1)))

df_sub = pd.DataFrame({'id': test_id, 'price_doc': pred_test[:, 0]})
df_sub.to_csv(
    '../output/stacking/Submission-SillyDataBaseLine-GunJa-Stacking2-2017060700-Test.csv', index=False)

df_oof = pd.DataFrame({'id': train_clean_id, 'price_doc': pred_oof[:, 0]})
df_oof.to_csv(
    '../output/stacking/Submission-SillyDataBaseLine-GunJa-Stacking2-20170606700-OutOfFold-ToBeFixed.csv', index=False)
