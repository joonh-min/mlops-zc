from typing import List, Tuple

import joblib
import mlflow
from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from homework_03.utils.encoders import vectorize_features
from homework_03.utils.feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_exporter
def export(
    data: Tuple[DataFrame, DataFrame], *args, **kwargs
) -> Tuple:

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("homework-03")

    df_train, df_val = data
    target = kwargs.get('target', 'duration')

    X_train, X_val, dv = vectorize_features(
        select_features(df_train),
        select_features(df_val),
    )
    y_train = df_train[target]
    y_val = df_val[target]

    with mlflow.start_run():
        mlflow.sklearn.autolog()

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        print(f"Intercept: {lr.intercept_}")

        y_pred = lr.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        joblib.dump(dv, 'dv.pkl')
        mlflow.log_artifact('dv.pkl')

    return lr, X_val, y_val, y_pred, rmse
