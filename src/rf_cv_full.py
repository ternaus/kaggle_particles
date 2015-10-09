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
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.cross_validation import ShuffleSplit
from sklearn.calibration import CalibratedClassifierCV

#import evaluation
exec(open("../../flavours-of-physics-start/evaluation.py").read())

#have some feature engineering work for better rank
# def add_features(df):
#     #significance of flight distance
#     df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    
#     return df

def add_features(df):
    #significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance'] / df['FlightDistanceError']
    #df['NEW']=df['IP']*df['dira']
    df['NEW_FD_SUMP']=df['FlightDistance'] / (df['p0_p'] + df['p1_p'] + df['p2_p'])
    df['NEW5_lt'] = df['LifeTime'] * (df['p0_IP'] + df['p1_IP'] + df['p2_IP']) / 3.0
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    return df


print("Load the training/test data using pandas")
train = pd.read_csv("../data/training.csv", na_values=[-99])
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
# filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal','SPDhits','IP', 'IPSig', 'isolatonic']
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal','SPDhits','p0_track_Chi2Dof','CDF1', 'CDF2', 'CDF3','isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta','DOCAone', 'DOCAtwo', 'DOCAthree']
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
  for n_estimators in [100]:
    for min_samples_split in [1]:
      for max_features in [0.7, 0.8]:
        for max_depth in [None]:
          for min_samples_leaf in [1]:
            score = []
                
            for train_index, test_index in rs:

              a_train = X.values[train_index]
              a_test = X.values[test_index]
              b_train = y.values[train_index]
              b_test = y.values[test_index]

              clf = RandomForestClassifier(n_estimators=n_estimators,
                        min_samples_split=min_samples_split,
                        max_features=max_features,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        n_jobs=2,
                        random_state=random_state,
                        criterion='entropy')

              clf.fit(a_train, b_train)

              preds = clf.predict_proba(a_test)[:, 1]

              # print clf.predict( xgb.DMatrix(check_agreement[features].values) )[:10]
              agreement_probs = clf.predict_proba(check_agreement[features])[:, 1]

              ks = compute_ks(
                      agreement_probs[check_agreement['signal'].values == 0],
                      agreement_probs[check_agreement['signal'].values == 1],
                      check_agreement[check_agreement['signal'] == 0]['weight'].values,
                      check_agreement[check_agreement['signal'] == 1]['weight'].values)
              print ('KS metric', ks, ks < 0.09)
              if ks >= 0.09:
                sys.exit()

              correlation_probs = clf.predict_proba(check_correlation[features])[:, 1]
              print ('Checking correlation...')
              cvm = compute_cvm(correlation_probs, check_correlation['mass'])
              print ('CvM metric', cvm, cvm < 0.002)

              # train_eval_probs = clf.predict(xgb.DMatrix(train_eval[features].values))
              train_eval_probs = clf.predict_proba(a_test)[:, 1]
              print ('Calculating AUC...')
              AUC = roc_auc_truncated(b_test, train_eval_probs)

              
              
              score += [AUC]
              print AUC

            sc = math.ceil(100000 * np.mean(score)) / 100000
            sc_std = math.ceil(100000 * np.std(score)) / 100000

            
            x = (sc,
               sc_std,
               n_estimators,
               min_samples_split,
               min_samples_leaf,
               max_depth,
               max_features,
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

  clf = RandomForestClassifier(n_estimators=800,
                        min_samples_split=1,
                        max_features=0.9,
                        max_depth=None,
                        min_samples_leaf=1,
                        n_jobs=1,
                        random_state=random_state,
                        criterion='entropy')
  
  rs = StratifiedKFold(train['signal'], n_folds=5, shuffle=True, random_state=random_state)

  clf = CalibratedClassifierCV(base_estimator=clf, method='isotonic', cv=rs)

  clf.fit(X, y)
  agreement_probs = clf.predict_proba(check_agreement[features])[:, 1]

  ks = compute_ks(
          agreement_probs[check_agreement['signal'].values == 0],
          agreement_probs[check_agreement['signal'].values == 1],
          check_agreement[check_agreement['signal'] == 0]['weight'].values,
          check_agreement[check_agreement['signal'] == 1]['weight'].values)
  print ('KS metric', ks, ks < 0.09)

  correlation_probs = clf.predict_proba(check_correlation[features])[:, 1]
  print ('Checking correlation...')
  cvm = compute_cvm(correlation_probs, check_correlation['mass'])
  print ('CvM metric', cvm, cvm < 0.002)

  prediction = clf.predict_proba(X_test)[:, 1]
  submission = pd.DataFrame()
  submission['id'] = test['id']
  submission['prediction'] = prediction
  submission.loc[X_test['LifeTime'] < 0, 'prediction'] = 0
  submission.loc[X_test['FlightDistance'] < 0, 'prediction'] = 0
  submission.to_csv("predictions/RF_calib_800.csv", index=False)

