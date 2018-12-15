import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold

from utils import load_datasets, load_target
from logs.logger import log_best
from models.lgbm import train_and_predict_lgbm


now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now))

feats = [
    'date_cos',
    'date_sin',
    'day_cos',
    'day_sin',
    'dow_cos',
    'dow_sin',
    'edited_dow',
    'dow',
    'month_cos',
    'month_sin',
    'month',
    'year',
    'holiday',
    'edited_holiday',
    'park_dummies',
    'agg_dow_park',
    'agg_dow_month',
    'agg_park_month',
    'agg_dow_park_month',
    'agg_dow',
    'agg_park',
    'agg_month',
    'weather',
    'lat_lon',
    'hotlink',
    'tommorow_holiday',
    'yesterday_holiday',
    'access_date'
]

logging.debug(feats)

X_train_all, X_test = load_datasets(feats)
y_train_all = load_target()
logging.debug(X_train_all.shape)

y_preds = []
models = []

kf = KFold(n_splits=8, random_state=0)
for train_index, valid_index in kf.split(X_train_all):
    X_train, X_valid = (
        X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :]
    )
    y_train, y_valid = y_train_all[train_index], y_train_all[valid_index]

    # lgbmの実行
    y_pred, model = train_and_predict_lgbm(
        X_train, X_valid, y_train, y_valid, X_test
    )

    # 結果の保存
    y_preds.append(y_pred)
    models.append(model)

    # スコア
    log_best(model)

# CVスコア
scores = [m.best_score['valid_0']['l1'] for m in models]
score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)
logging.debug('===CV scores===')
logging.debug(scores)
logging.debug(score)

# submitファイルの作成
test = pd.read_feather('./data/input/test.feather')
sub = [pd.DataFrame(df, index=test.index) for df in y_preds]

for i in range(len(sub) - 1):
    sub[0].loc[:, 0] += sub[i + 1].loc[:, 0]
sub[0].loc[:, 0] /= len(sub)

sub[0].to_csv(
    './data/output/sub_{0:%Y%m%d%H%M%S}_{1}.tsv'.format(now, score),
    sep='\t', header=None
)
