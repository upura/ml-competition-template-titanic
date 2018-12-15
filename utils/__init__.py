import time
import pandas as pd
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_datasets(feats):
    dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test


def load_target():
    train = pd.read_feather('./data/input/train.feather')
    y_train = train['visitors']
    return y_train
