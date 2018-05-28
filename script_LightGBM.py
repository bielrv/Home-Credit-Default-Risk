import pandas as pd
# pandas is a library for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series

import numpy as np
# numpy adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
# sklearn features various classification, regression and clustering algorithms including support vector machines
# sklearn.metrics module includes score functions, performance metrics and pairwise metrics and distance computations
# sklearn.metrics.roc_auc_score computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
# sklearn.metrics.precision_recall_curve computes precision-recall pairs for different probability thresholds
# sklearn.metrics.roc_curve computes Receiver Operating Characteristic (ROC)

# In statistics, a Receiver Operating Characteristic curve, i.e. ROC curve, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

from sklearn.model_selection import KFold
# sklearn.model_selection.KFold provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds

import matplotlib.pyplot as plt
# matplotlib.pyplot is a collection of command style functions that make matplotlib work like MATLAB

import seaborn as sns
# Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.

import gc
# This module provides an interface to the optional garbage collector. It provides the ability to disable the collector, tune the collection frequency, and set debugging options.

gc.enable()
# Enable automatic garbage collection.

#-------------------------------------
# Step 1 - Including bureau_balance information
#-------------------------------------

# Load bureau_balance.csv
buro_bal = pd.read_csv('../input/bureau_balance.csv')
# Read CSV (comma-separated) file into DataFrame
print('Buro bal shape : ', buro_bal.shape)
# Buro bal shape :  (27299925, 3)
type(print)

# dummies is very useful for categorical variables
print('transform to dummies')
buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS', axis=1)
# pandas.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
# pandas.concat concatenates pandas objects along a particular axis with optional set logic along the other axes.

# pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)
# Convert categorical variable into dummy/indicator variables: creates one column for each diferent value in data
# String to append DataFrame column names Pass a list with length equal to the number of columns when calling get_dummies on a DataFrame.
buro_bal.columns

print('Counting month buros')
buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
# DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False, observed=False, **kwargs)
# DataFrame.groupby groups series using mapper (dict or key function, apply given function to group, return result as series) or by a series of columns.
# DataFrame.count(axis=0, level=None, numeric_only=False)
# DataFrame.count counts non-NA cells for each column or row.

# Check buro_counts structure
buro_counts.head(10)
buro_counts.columns

buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])
# Map values of Series using input correspondence (which can be a dict, Series, or function)
# Appends buro_counts dataframe to buro_bal dataframe by SK_ID_BUREAU

# Check buro_bal structure
buro_bal.head(10)
buro_bal.columns

print('Averaging buro bal')
avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()
# DataFrame.groupby groups series using mapper
# DataFrame.mean calculates the mean of that field

avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]

# Delete auxiliary table buro_bal
del buro_bal

# reclaim all memory that is inaccessible
gc.collect()
# With no arguments, run a full collection.
# In computer science, garbage collection (GC) is a form of automatic memory management. 
# The garbage collector, or just collector, attempts to reclaim garbage, or memory occupied by objects that are no longer in use by the program

avg_buro_bal.columns

#-------------------------------------
# Step 2 - Including bureau information
#-------------------------------------

print('Read Bureau')
# load bureau.csv 
buro = pd.read_csv('../input/bureau.csv')


print('Go to dummies')

# turns all the categorical variables into useful variables
buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
# Active, Bad debt, Closed, Sold
buro_credit_active_dum.columns

buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
buro_credit_currency_dum.columns
# Currency 1, Currency 2, Currency 3, Currency 4

buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')
buro_credit_type_dum.columns
# ...Car loan, Credit card, Microloan,...

# Append everything together
buro_full = pd.concat([buro, buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum], axis=1)
# buro_full.columns = ['buro_' + f_ for f_ in buro_full.columns]
buro_full.columns

# Delete auxiliary tables buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
# reclaim all memory that is inaccessible
gc.collect()

print('Merge with buro avg')
buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', on='SK_ID_BUREAU', suffixes=('', '_bur_bal'))

# DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
# Merge DataFrame objects by performing a database-style join operation by columns or indexes.
# If joining columns on columns, the DataFrame indexes will be ignored. Otherwise if joining indexes on indexes or indexes on a column or columns, the index will be passed on.


print('Counting buro per SK_ID_CURR')
# counts number of buros per current loan (SK_ID_CURR)
nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
# Maps buro_full and nb_bureau_per_curr
buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

print('Averaging bureau')
# Groups by current loan calculating mean 
avg_buro = buro_full.groupby('SK_ID_CURR').mean()
print(avg_buro.head())

# delete auxiliary tables
del buro, buro_full
# this method reclaims all memory that is inaccessible. It performs a blocking garbage collection of all generations.
gc.collect()

#-------------------------------------
# Step 3 - Including previous_application information
#-------------------------------------

print('Read prev')
# load previous_application.csv
prev = pd.read_csv('../input/previous_application.csv')

# find all categorical features in previous_application.csv
prev_cat_features = [
    f_ for f_ in prev.columns if prev[f_].dtype == 'object'
]
prev_cat_features

print('Go to dummies')
# Create new dataframe
prev_dum = pd.DataFrame()
# for each categorical feature (for f_ in prev_cat_features) concatanate dummies using prefix f_
for f_ in prev_cat_features:
    prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)

# prev.columns

len(prev_dum.columns) # 143 colummns


# prev.head(5)
# prev_dum.head(5)

# concatante prev_dum 
prev = pd.concat([prev, prev_dum], axis=1)

# len(prev.columns) # 180 columns


# delete auxiliary table prev_dum
del prev_dum
# reclaim all memory that is inaccessible
gc.collect()

print('Counting number of Prevs')
# counting prior loans by current loan
nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()

# check nb_prev_per_curr
type(nb_prev_per_curr)
nb_prev_per_curr.head(3)
nb_prev_per_curr.shape

# append nb of prev loans in nb_prev_per_curr df by curr loan id
prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])

len(prev.columns) # 180 - SK_ID_PREV now contains nb of prev loans
prev['SK_ID_PREV'].head(3)
prev.shape # (1670214, 180)

# print Averaging prev
print('Averaging prev')
# calculate means in prev datarfame grouping by current loan - TIME CONSUMING STEP
avg_prev = prev.groupby('SK_ID_CURR').mean()
# print avg_prev df head
print(avg_prev.head())

avg_prev.shape # (X, X)

# delete auxiliary table prev
del prev
# reclaim all memory that is inaccessible
gc.collect()

#-------------------------------------
# Step 4 - Including POS_CASH_balance information
#-------------------------------------

# print Reading POS_CASH
print('Reading POS_CASH')
# load POS_CASH_balance.csv into pos dataframe
pos = pd.read_csv('../input/POS_CASH_balance.csv')

# print Go to dummies
print('Go to dummies')
# concatanate pos[NAME_CONTRACT_STATUS] dummies in pos dataframe
pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)

# print Compute nb of prevs per curr
print('Compute nb of prevs per curr')
# count number of prior loans in nb_prevs grouping by current loan
nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
# append number of prior loans
pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

# pos.columns

# print Go to averages
print('Go to averages')
# calculate the mean grouping by current loan of all fields in avg_pos dataframe
avg_pos = pos.groupby('SK_ID_CURR').mean()

# delete auxiliary tables pos, nb_prevs
del pos, nb_prevs
# reclaim all memory that is inaccessible
gc.collect()

#-------------------------------------
# Step 5 - Including credit_card_balance information
#-------------------------------------

# print Reading CC balance
print('Reading CC balance')
# load credit_card_balance.csv to cc_bal dataframe
cc_bal = pd.read_csv('../input/credit_card_balance.csv')

# print Go to dummies
print('Go to dummies')
# append in cc_bal dataframe NAME_CONTRACT_STATUS dummies using cc_bal_status_ prefix
cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')], axis=1)

nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

# print Compute average
print('Compute average')
# calculate all means and group by current loan
avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
# placing cc_bal_ prefix in avg_cc_bal dataframe columns
avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]

# delete auxiliary cc_bal, nb_prevs
del cc_bal, nb_prevs
# reclaim all memory that is inaccessible
gc.collect()

#-------------------------------------
# Step 6 - Including installments_payments information
#-------------------------------------

print('Reading Installments')
inst = pd.read_csv('../input/installments_payments.csv')
nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

avg_inst = inst.groupby('SK_ID_CURR').mean()
avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]

#-------------------------------------
# Step 7 - Including application_train information
#-------------------------------------

print('Read data and test')
data = pd.read_csv('../input/application_train.csv')

test = pd.read_csv('../input/application_test.csv')
print('Shapes : ', data.shape, test.shape)

# moving TARGET series from data df to y series
y = data['TARGET']
del data['TARGET']

# identify categorical features in application_train into categorical_feats series
categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]

# checking categorical_feats
# categorical_feats

for f_ in categorical_feats:
# for each categorical value

    data[f_], indexer = pd.factorize(data[f_])
# encode the object as an enumerated type or categorical variable.
# this method is useful for obtaining a numeric representation of an array when all that matters is identifying distinct values.
# indexer contains labels used to factorize

    test[f_] = indexer.get_indexer(test[f_])
# Compute indexer and mask for new index given the current index. The indexer should be then used as an input to ndarray.take to align the current data to the new index.
    
# 1. Append aggregated buro data from avg_buro df
data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

# 2. Append aggregated previous loans data from avg_prev df
data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

# 3. Append aggregated Monthly balance from avg_pos df
data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')

# 4. Append aggregated credit card balances from avg_cc_bal df
data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

# 5. Append aggregated installments data from avg_inst df
data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

import gc
gc.enable()

# delete auxiliary tables avg_buro, avg_prev 
del avg_buro, avg_prev
# reclaim all memory that is inaccessible
gc.collect()

from lightgbm import LGBMClassifier
# A fast, distributed, high performance gradient boosting (GBDT, GBRT, GBM or MART) framework based on decision tree algorithms  
# used for ranking classification and many other machine learning tasks. It is under the umbrella of the DMTK

# KFold(n_splits=3, shuffle=False, random_state=None)
# K-Folds cross-validator
# Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default)
# Each fold is then used once as a validation while the k - 1 remaining folds form the training set


folds = KFold(n_splits=5, shuffle=True, random_state=546789)

# create oof_preds array of shape [0] filled with zeros
oof_preds = np.zeros(data.shape[0])
# create sub_preds array of shape [0] filled with zeros
sub_preds = np.zeros(test.shape[0])

# create feature_importance_df dataframe
feature_importance_df = pd.DataFrame()

# create feats serie containing all features 
feats = [f for f in data.columns if f not in ['SK_ID_CURR']]

# enumarate is a built-in function of Python which allows us to loop over something and have an automatic counter
# folds.split generates indices to split data into training and test set as set-up in folds
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]

# Light GBM is a gradient boosting framework that uses tree based learning algorithm. 
    clf = LGBMClassifier(
        # n_estimators=1000,
        # num_leaves=20,
        # colsample_bytree=.8,
        # subsample=.8,
        # max_depth=7,
        # reg_alpha=.1,
        # reg_lambda=.1,
        # min_split_gain=.01
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=30,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2,
        silent=-1,
        verbose=-1,
    )
    
    #  train clasifier
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    
    sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
    
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 

test['TARGET'] = sub_preds

test[['SK_ID_CURR', 'TARGET']].to_csv('first_submission.csv', index=False)

# Plot feature importances
cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(8,10))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

# Plot ROC curves
plt.figure(figsize=(6,6))
scores = [] 
for n_fold, (_, val_idx) in enumerate(folds.split(data)):  
    # Plot the roc curve
    fpr, tpr, thresholds = roc_curve(y.iloc[val_idx], oof_preds[val_idx])
    score = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
    scores.append(score)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
fpr, tpr, thresholds = roc_curve(y, oof_preds)
score = roc_auc_score(y, oof_preds)
plt.plot(fpr, tpr, color='b',
         label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LightGBM ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig('roc_curve.png')

# Plot ROC curves
plt.figure(figsize=(6,6))
precision, recall, thresholds = precision_recall_curve(y, oof_preds)
score = roc_auc_score(y, oof_preds)
plt.plot(precision, recall, color='b',
         label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('LightGBM Recall / Precision')
plt.legend(loc="best")
plt.tight_layout()

plt.savefig('recall_precision_curve.png')

