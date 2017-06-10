# coding=utf-8

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import model_selection, preprocessing
# import xgboost as xgb
# import datetime

louis = pd.read_csv(
    '../output/single/Submission-SillyGame-Louis.csv', index_col='id')
bruno = pd.read_csv(
    '../output/single/Submission-SillyGame-Bruno.csv', index_col='id')
gunja = pd.read_csv(
    '../output/single/Submission-SillyGame-GunJa.csv', index_col='id')
gunja_stacking = pd.read_csv(
    '../output/stacking/Submission-SillyDataBaseLine-GunJa-Stacking2-2017060900-Test.csv', index_col='id')

merge = pd.DataFrame(
    {
        'louis': louis.price_doc,
        'bruno': bruno.price_doc,
        'gunja': gunja.price_doc,
        'gunja_stacking': gunja_stacking.price_doc
    }
)


merge['follow'] = np.exp(
    .714 * np.log(merge.louis) +
    .286 * np.log(merge.bruno)
)
merge['price_doc_suba'] = np.exp(
    .7 * np.log(merge.follow) +
    .3 * np.log(merge.gunja)
)
merge['price_doc_subb'] = np.exp(
    .7 * np.log(merge.follow) +
    .3 * np.log(merge.gunja_stacking)
)
merge['price_doc_subc'] = merge.price_doc_suba * \
    0.5 + merge.price_doc_subb * 0.5


def save_submission(df, column, tag):
    pd.DataFrame(
        {
            'price_doc': df[column]
        }
    ).to_csv(
        '../output/ensemble/Submission-Ensemble-SillyGame-{}.csv'.format(tag)
    )


save_submission(merge, 'price_doc_suba', 'Origin-2017061000')
save_submission(merge, 'price_doc_subb', 'Stacking2-2017061000')
save_submission(merge, 'price_doc_subc', 'Average-2017061000')
