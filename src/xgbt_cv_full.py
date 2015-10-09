from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
3 days left. I do not have any bright ideas right now, so I will try to
find single model with best parameters and average with other predictors
'''
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
import numpy as np
from sklearn.cross_validation import *
import sys

import math
from sklearn.cross_validation import ShuffleSplit
#import evaluation
exec(open("../../flavours-of-physics-start/evaluation.py").read())

#have some feature engineering work for better rank
def add_features(df):
    #significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    return df

print("Load the training/test data using pandas")
train = pd.read_csv("../data/training.csv")
# test = pd.read_csv("../data/test.csv")

print("Adding features to both training and testing")
train = add_features(train)
# test = add_features(test)

print("Loading check agreement for KS test evaluation")
check_agreement = pd.read_csv('../data/check_agreement.csv')
check_correlation = pd.read_csv('../data/check_correlation.csv')
check_agreement = add_features(check_agreement)
check_correlation = add_features(check_correlation)

# train_eval = train[train['min_ANNmuon'] > 0.4]

print train.columns
print("Eliminate SPDhits, which makes the agreement check fail")
# filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal','SPDhits','IP', 'IPSig', 'isolatonic']
# filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal','SPDhits','IP', 'IPSig', 'isolatonic']
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal','SPDhits','IP', 'IPSig', 'isolatonic']
features = list(f for f in train.columns if f not in filter_out)
print("features:",features)

print("Train a XGBoost model")


y = train['signal']

X = train[features]
# X_test = test[features]

params = {
  # 'objective': 'reg:linear',
  # 'objective': 'count:poisson',
  'objective': 'binary:logistic',
  # 'eta': 0.005,
  # 'min_child_weight': 6,
  # 'subsample': 0.7,
  # 'colsabsample_bytree': 0.7,
  # 'scal_pos_weight': 1,
  'silent': 1,
  # 'max_depth': 9
}

num_rounds = 1000
random_state = 42
offset = 10000
test_size = 0.2


ind = 1
if ind == 1:
  n_iter = 5
  # rs = ShuffleSplit(len(y), n_iter=n_iter, test_size=test_size, random_state=random_state)
  rs = StratifiedKFold(train['signal'], n_folds=n_iter, shuffle=True, random_state=random_state)

  result = []
  for scale_pos_weight in [5]:
    for min_child_weight in [10]:
      for eta in [1]:
        for colsample_bytree in [0.7]:
          for max_depth in [4]:
            for subsample in [0.7]:
              for gamma in [1]:
                params['min_child_weight'] = min_child_weight
                params['eta'] = eta
                params['colsample_bytree'] = colsample_bytree
                params['max_depth'] = max_depth
                params['subsample'] = subsample
                params['gamma'] = gamma
                params['scale_pos_weight'] = scale_pos_weight

                params_new = list(params.items())
                score = []
                # score_truncated_up = []
                # score_truncated_down = []
                score_truncated_both = []
                # score_truncated_both_round = []
                # score_truncated_both_int = []

                for train_index, test_index in rs:

                  X_train = X.values[train_index]
                  X_test = X.values[test_index]
                  y_train = y.values[train_index]
                  y_test = y.values[test_index]

                  xgtest = xgb.DMatrix(X_test)

                  xgtrain = xgb.DMatrix(X_train[offset:, :], label=y_train[offset:])
                  xgval = xgb.DMatrix(X_train[:offset, :], label=y_train[:offset])

                  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                  clf = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

                  preds1 = clf.predict(xgtest, ntree_limit=clf.best_iteration)

                  # X_train = X_train[::-1, :]
                  # labels = y_train[::-1]

                  # xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
                  # xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

                  # watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                  # model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

                  # preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)

                  # preds = 0.5 * preds1 + 0.5 * preds2
                  preds = preds1

                  # print clf.predict( xgb.DMatrix(check_agreement[features].values) )[:10]
                  agreement_probs = clf.predict( xgb.DMatrix(check_agreement[features].values))

                  ks = compute_ks(
                          agreement_probs[check_agreement['signal'].values == 0],
                          agreement_probs[check_agreement['signal'].values == 1],
                          check_agreement[check_agreement['signal'] == 0]['weight'].values,
                          check_agreement[check_agreement['signal'] == 1]['weight'].values)
                  print ('KS metric', ks, ks < 0.09)
                  if ks >= 0.09:
                    sys.exit()

                  correlation_probs = clf.predict(xgb.DMatrix(check_correlation[features].values))
                  print ('Checking correlation...')
                  cvm = compute_cvm(correlation_probs, check_correlation['mass'])
                  print ('CvM metric', cvm, cvm < 0.002)

                  # train_eval_probs = clf.predict(xgb.DMatrix(train_eval[features].values))
                  train_eval_probs = clf.predict(xgb.DMatrix(X_test))
                  print ('Calculating AUC...')
                  AUC = roc_auc_truncated(y_test, train_eval_probs)

                  
                  
                  score += [AUC]
                  print AUC

                sc = math.ceil(100000 * np.mean(score)) / 100000
                sc_std = math.ceil(100000 * np.std(score)) / 100000
                result += [(sc,
                            sc_std,
                            min_child_weight,
                            eta,
                            colsample_bytree,
                            max_depth,
                            subsample,
                            gamma,
                            n_iter,
                            params['objective'],
                            test_size,
                            scale_pos_weight)]

    result.sort()

    print
    print 'result'
    print result


elif ind == 2:
  test = pd.read_csv("../data/test.csv")
  test = add_features(test)
  X_test = test[features]

  xgtrain = xgb.DMatrix(X.values[offset:, :], label=y.values[offset:])
  xgval = xgb.DMatrix(X.values[:offset, :], label=y.values[:offset])
  xgtest = xgb.DMatrix(X_test.values)

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  params = {
  # 'objective': 'reg:linear',
    # 'objective': 'count:poisson',
      'objective': 'binary:logistic',
  'eta': 0.1,
  'min_child_weight': 1,
  'subsample': 1,
  'colsample_bytree': 0.5,
  # 'scal_pos_weight': 1,
  'silent': 1,
  'max_depth': 10,
  'gamma': 1
  }    
  params_new = list(params.items())
  model1 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
  
  agreement_probs = clf.predict( xgb.DMatrix(check_agreement[features].values))

  ks = compute_ks(
          agreement_probs[check_agreement['signal'].values == 0],
          agreement_probs[check_agreement['signal'].values == 1],
          check_agreement[check_agreement['signal'] == 0]['weight'].values,
          check_agreement[check_agreement['signal'] == 1]['weight'].values)
  print ('KS metric', ks, ks < 0.09)

  correlation_probs = model1.predict(xgb.DMatrix(check_correlation[features].values))
  print ('Checking correlation...')
  cvm = compute_cvm(correlation_probs, check_correlation['mass'])
  print ('CvM metric', cvm, cvm < 0.002)

  prediction_test_1 = model1.predict(xgtest, ntree_limit=model1.best_iteration)

  X_train = X.values[::-1, :]
  labels = y.values[::-1]

  xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
  xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  model2 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
  agreement_probs = model2.predict( xgb.DMatrix(check_agreement[features].values))

  ks = compute_ks(
          agreement_probs[check_agreement['signal'].values == 0],
          agreement_probs[check_agreement['signal'].values == 1],
          check_agreement[check_agreement['signal'] == 0]['weight'].values,
          check_agreement[check_agreement['signal'] == 1]['weight'].values)
  print ('KS metric', ks, ks < 0.09)

  correlation_probs = model2.predict(xgb.DMatrix(check_correlation[features].values))
  print ('Checking correlation...')
  cvm = compute_cvm(correlation_probs, check_correlation['mass'])
  print ('CvM metric', cvm, cvm < 0.002)


  prediction_test_2 = model2.predict(xgtest, ntree_limit=model2.best_iteration)


  prediction_test = 0.5 * prediction_test_1 + 0.5 * prediction_test_2
  submission = pd.DataFrame()
  submission['id'] = test['id']
  submission['predictions'] = prediction_test
  submission.to_csv("predictions/xgbt.csv", index=False)

