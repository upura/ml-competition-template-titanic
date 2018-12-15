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
    'age',
    'embarked',
    'family_size',
    'fare',
    'pclass',
    'sex'
]

logging.debug(feats)

X_train_all, X_test = load_datasets(feats)
y_train_all = load_target()
logging.debug(X_train_all.shape)

y_preds = []
models = []

kf = KFold(n_splits=3, random_state=0)
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
scores = [m.best_score['valid_0']['multi_logloss'] for m in models]
score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)
logging.debug('===CV scores===')
logging.debug(scores)
logging.debug(score)

# submitファイルの作成
sub = pd.DataFrame(pd.read_csv('./data/input/test.csv')['PassengerId'])

for i in range(len(y_preds) - 1):
    y_preds[0] += y_preds[i + 1]

sub['Survived'] = [1 if y > int(len(y_pred)/2) else 0 for y in y_preds[0]]

sub.to_csv(
    './data/output/sub_{0:%Y%m%d%H%M%S}_{1}.csv'.format(now, score),
    index=False
)
