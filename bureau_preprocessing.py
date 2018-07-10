import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import warnings
from custom_modules.custom_modules import one_hot_encoder
warnings.simplefilter(action='ignore', category=FutureWarning)

debug = False
nan_as_category = True

num_rows = 10000 if debug else None

bureau = pd.read_csv('input/bureau.csv', nrows = num_rows)
print("Bureau df shape:", bureau.shape)

bb = pd.read_csv('input/bureau_balance.csv', nrows = num_rows)
print("Bureau balance df shape:", bb.shape)
bb.head(3)

bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
bb, bb_cat = one_hot_encoder(bb, nan_as_category)

bureau.head(10)

bb.head(10)

bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
for col in bb_cat:
    bb_aggregations[col] = ['mean']
bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
del bb, bb_agg
gc.collect()

bureau.head(10)
