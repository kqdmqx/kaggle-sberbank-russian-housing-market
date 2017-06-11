# coding=utf-8

import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from my_py_models.config import OUTPUT_PATH
from os.path import join

# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import model_selection, preprocessing
# import xgboost as xgb
# import datetime

louis = pd.read_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-Louis-Stacking2-2017061100-Test.csv'), index_col='id')
bruno = pd.read_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyGame-Bruno-Stacking2-2017061100-Test.csv'), index_col='id')
gunja = pd.read_csv(join(OUTPUT_PATH, 'stacking/Submission-SillyDataBaseLine-GunJa-Stacking2-2017060900-Test.csv'), index_col='id')

merge = pd.DataFrame(
    {
        'louis': louis.price_doc,
        'bruno': bruno.price_doc,
        'gunja': gunja.price_doc
    }
)


merge['follow'] = np.exp(
    .714 * np.log(merge.louis) +
    .286 * np.log(merge.bruno)
)
merge['price_doc'] = np.exp(
    .7 * np.log(merge.follow) +
    .3 * np.log(merge.gunja)
)


def save_submission(df, column, tag):
    pd.DataFrame(
        {
            'price_doc': df[column]
        }
    ).to_csv(
        join(OUTPUT_PATH, 'ensemble/Submission-Ensemble-SillyGameCombine-{}.csv'.format(tag))
    )


save_submission(merge, 'price_doc', 'TotalStacking-2017061100')
