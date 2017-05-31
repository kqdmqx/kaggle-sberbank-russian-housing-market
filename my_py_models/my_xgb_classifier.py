# coding=utf-8

import xgboost as xgb


class MyXgbClassifier:
    def __init__(self, params):
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        dtrain = xgb.DMatrix(X_train, y_train)
        dval = xgb.DMatrix(X_val, y_val)

        partial_model = xgb.train(self.params, dtrain, evals=[(dval, 'val')],
                                  num_boost_round=1000,
                                  early_stopping_rounds=25, verbose_eval=25)
        num_boost_round = partial_model.best_iteration

        self.model = xgb.train(dict(self.params, silent=0), dtrain,
                               num_boost_round=num_boost_round)

    def predict(self, X_test):
        if self.model is None:
            return None
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)
