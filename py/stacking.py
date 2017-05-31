# coding=utf-8

import numpy as np
from sklearn.cross_validation import KFold


class Stacking:
    def __init__(self, n_folds, base_models):
        '''
        Ensemble Stacking Level
        Parameters:
            param1: self
            param2: cv ford num
            param3: list of clf
        '''
        self.n_folds = n_folds
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        '''
        Foreach base_model: fit X, y
        Parameters:
            param1: self
            param2：X, ndarray, features of train set
            param3: y, ndarray, target values of train set
            param4: X_test, ndarray, features of test set
        Return:
            out_of_fold_pred: ndarray, X.shape[0] * len(base_models)
            test_pred: ndarray, T.shape[0] * len(base_models)

        '''
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds,
                           shuffle=True, random_state=2016))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(1)

        out_of_fold_pred = S_train
        test_pred = S_test
        return out_of_fold_pred, test_pred
