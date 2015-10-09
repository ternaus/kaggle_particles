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
# from sklearn.ensemble import RandomForestClassifier
import graphlab as gl
import math
from sklearn.cross_validation import ShuffleSplit
#import evaluation
exec(open("../../flavours-of-physics-start/evaluation.py").read())

#have some feature engineering work for better rank
def add_features(df):
    #significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance'] / df['FlightDistanceError']
    return df

print("Load the training/test data using pandas")
train = pd.read_csv("../data/training.csv", na_values=[-99])
# test = pd.read_csv("../data/test.csv")

print("Adding features to both training and testing")
train = add_features(train)
# test = add_features(test)

print("Loading check agreement for KS test evaluation")
check_agreement = gl.SFrame('../data/check_agreement.csv')

check_correlation = gl.SFrame('../data/check_correlation.csv')

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

print("Train a RF model")


y = train['signal']

X = train[features]
# X_test = test[features]


num_rounds = 1000
random_state = 42
offset = 10000
test_size = 0.2


ind = 1
if ind == 1:
  n_iter = 3
  # rs = ShuffleSplit(len(y), n_iter=n_iter, test_size=test_size, random_state=random_state)
  rs = StratifiedKFold(train['signal'], n_folds=n_iter, shuffle=True, random_state=random_state)

  result = []
  for num_trees in [100]:
    for step_size in [None]:
      for min_loss_reduction in [None]:
        for min_child_weight in [None]:
          for max_depth in [16, 18]:
            for row_subsample in [None]:
              for column_subsample in [0.9]:
            
                score = []
                    
                for train_index, test_index in rs:

                  a_train = X.values[train_index]
                  a_test = X.values[test_index]
                  b_train = y.values[train_index]
                  b_test = y.values[test_index]
                  
                  a_train = pd.DataFrame(a_train)
                  a_train.columns = features
                  a_train = gl.SFrame(a_train)

                  a_test = pd.DataFrame(a_test)
                  a_test.columns = features
                  a_test = gl.SFrame(a_test)
                  
                  a_train['target'] = b_train
                  a_test['target'] = b_test

                  # clf = RandomForestClassifier(n_estimators=n_estimators,
                  #           min_samples_split=min_samples_split,
                  #           max_features=max_features,
                  #           max_depth=max_depth,
                  #           min_samples_leaf=min_samples_leaf,
                  #           n_jobs=-1,
                  #           random_state=random_state,
                  #           criterion='entropy')
                  clf = gl.random_forest_classifier.create(a_train, 
                    target='target',
                    features=features,
                    num_trees=num_trees,
                    max_depth=max_depth,
                    step_size=step_size,
                    min_loss_reduction=min_loss_reduction,
                    min_child_weight=min_child_weight,
                    row_subsample=row_subsample,
                    column_subsample=column_subsample,
                    validation_set=None,
                    random_seed=random_state,
                    verbose=False
                    )

                  # clf.fit(a_train, b_train)

                  preds = clf.predict(a_test, output_type='probability')

                  # print clf.predict( xgb.DMatrix(check_agreement[features].values) )[:10]
                  ch = gl.SFrame(check_agreement[features])


                  agreement_probs = clf.predict(ch, output_type='probability')

                  ks = compute_ks(
                          agreement_probs[check_agreement['signal'] == 0],
                          agreement_probs[check_agreement['signal'] == 1],
                          check_agreement[check_agreement['signal'] == 0]['weight'],
                          check_agreement[check_agreement['signal'] == 1]['weight'])
                  # print ('KS metric', ks, ks < 0.09)
                  if ks >= 0.09:
                    sys.exit()

                  correlation_probs = clf.predict(check_correlation[features], output_type='probability')
                  # print ('Checking correlation...')
                  cvm = compute_cvm(correlation_probs, check_correlation['mass'])
                  # print ('CvM metric', cvm, cvm < 0.002)

                  # train_eval_probs = clf.predict(xgb.DMatrix(train_eval[features].values))
                  train_eval_probs = clf.predict(a_test, output_type='probability')
                  # print ('Calculating AUC...')
                  AUC = roc_auc_truncated(b_test, train_eval_probs)

                  
                  
                  score += [AUC]
                  # print AUC

                sc = math.ceil(100000 * np.mean(score)) / 100000
                sc_std = math.ceil(100000 * np.std(score)) / 100000

                
                x = (sc,
                   sc_std,
                   num_trees,
                   max_depth,
                   step_size,
                   min_loss_reduction,
                   min_child_weight,
                   row_subsample,
                   column_subsample,                   
                   n_iter,
                   test_size)
                print x
              result += [x]

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

