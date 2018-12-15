import pandas as pd


target = [
    'park'
]

for t in target:
    train = pd.read_feather('./features/' + t + '_train.feather')
    test = pd.read_feather('./features/' + t + '_test.feather')

    train = pd.get_dummies(train, prefix='park', columns=['park'])
    test = pd.get_dummies(test, prefix='park', columns=['park'])

    train.to_feather('./features/' + t + '_dummies_train.feather')
    test.to_feather('./features/' + t + '_dummies_test.feather')
