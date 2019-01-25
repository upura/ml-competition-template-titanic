import numpy as np
import catboost


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test):

    categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
    print(X_train.head())
    print(categorical_features_indices)

    model = catboost.CatBoostClassifier(
        iterations=10, 
        use_best_model=True,
        eval_metric = 'Accuracy'
    )

    model.fit(
        X_train, y_train, 
        cat_features=categorical_features_indices,
        eval_set=(X_valid, y_valid), 
        plot=True
    )

    # テストデータを予測する
    y_pred = model.predict(X_test)

    return y_pred, model
