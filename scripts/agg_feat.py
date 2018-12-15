import pandas as pd
from utils import load_datasets, load_target


def create_agg_feat(keys):
    feats = keys + ['datetime']
    train, test = load_datasets(feats)
    train['visitors'] = load_target()

    agg = train.groupby(keys).agg(aggs).reset_index()
    agg_names = ['_'.join(keys) + '_' + a for a in list(aggs)]
    agg.columns = (keys + agg_names)

    agg_train = pd.merge(train, agg, on=keys, how='left')
    agg_test = pd.merge(test, agg, on=keys, how='left')

    agg_train[agg_names].to_feather(
        './features/agg_' + '_'.join(keys) + '_train.feather'
    )
    agg_test[agg_names].to_feather(
        './features/agg_' + '_'.join(keys) + '_test.feather'
    )

    print('agg_' + '_'.join(keys))


aggs = {'nunique', 'count', 'sum', 'min', 'mean', 'median', 'max', 'var'}

create_agg_feat(['dow', 'park'])
create_agg_feat(['dow', 'month'])
create_agg_feat(['park', 'month'])
create_agg_feat(['dow', 'park', 'month'])
create_agg_feat(['dow'])
create_agg_feat(['park'])
create_agg_feat(['month'])
