# coding=utf-8
import pandas as pd
import numpy as np
import sys
sys.path.append('../..')
from my_py_models.stacking import Stacking
from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from os.path import join, isfile
from sklearn.metrics import mean_squared_error


# 读取已知的lb_score
with open(join(OUTPUT_PATH, 'analysis/submission-history-20170614.txt')) as ifile:
    lines = ifile.readlines()

lines = filter(lambda x: x.startswith('0') or
               x.startswith('Submission'), lines)
lines = map(lambda x: x.strip(), lines)

file_names = map(lambda x: x[1],
                 filter(lambda x: x[0] % 2 == 0,
                        enumerate(lines)))
lb_scores = map(lambda x: x[1],
                filter(lambda x: x[0] % 2 == 1,
                       enumerate(lines)))
stacking_subs = filter(lambda x: x[0].endswith('Test.csv'),
                       zip(file_names, lb_scores))
stacking_subs = map(lambda (x, y): (x, float(y)), stacking_subs)


# 读取目标函数
train = pd.read_csv(join(INPUT_PATH, 'train.csv'))
test = pd.read_csv(join(INPUT_PATH, 'test.csv'))

train_id = pd.read_csv(
    join(OUTPUT_PATH, 'sample/TrainId-SillyGame-Gunja-Clean.csv'))
train = train.loc[train.id.isin(train_id.id.unique())].copy()


def match_oof_file(filename):
    return filename.replace('Test', 'OutOfFold')


def match_pred(filename):
    temp = pd.read_csv(join(OUTPUT_PATH, 'stacking/' + filename))
    # temp = temp.loc[temp.id.isin(train_id.id.unique())]
    return pd.DataFrame(
        {
            'pred': temp.set_index('id').price_doc,
            'y_train': train.set_index('id').price_doc
        }
    )


print 'submission,lb,cv1,cv2'
for fn, lb in stacking_subs:
    match = match_pred(match_oof_file(fn))
    match = match.loc[match.index.isin(train_id.id.unique())]
    match.fillna(match.pred.mean(), inplace=True)
    cv1 = np.sqrt(mean_squared_error(np.log(match.pred + 1), np.log(match.y_train + 1)))
    cv2 = np.sqrt(mean_squared_error(np.log(match.pred + 1), np.log(match.y_train * .969 + 10)))
    print '{},{},{},{}'.format(fn, lb, cv1, cv2)
