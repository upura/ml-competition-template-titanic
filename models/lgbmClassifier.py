import lightgbm as lgb
import logging
import numpy as np

from logs.logger import log_evaluation


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, lgbm_params):

    # データセットを生成する
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    logging.debug(lgbm_params)

    # ロガーの作成
    logger = logging.getLogger('main')
    callbacks = [log_evaluation(logger, period=30)]

    # 上記のパラメータでモデルを学習する
    model = lgb.train(
        lgbm_params, lgb_train,
        # モデルの評価用データを渡す
        valid_sets=lgb_eval,
        # 最大で 1000 ラウンドまで学習する
        num_boost_round=1000,
        # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
        early_stopping_rounds=10,
        # ログ
        callbacks=callbacks
    )

    # テストデータを予測する
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_max = np.argmax(y_pred, axis=1)
    print(len(X_test))

    return y_pred_max, model
