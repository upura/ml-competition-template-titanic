import pandas as pd


def load_datasets(feats):
    dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1, sort=False)
    dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv('./data/input/train.csv')
    y_train = train[target_name]
    return y_train
