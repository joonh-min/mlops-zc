from typing import Tuple

import pandas as pd

from homework_03.utils.cleaning import clean
from homework_03.utils.feature_engineering import combine_features
from homework_03.utils.feature_selector import select_features
from homework_03.utils.splitters import split_on_value

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_on_feature = kwargs.get('split_on_feature', 'tpep_pickup_datetime')
    split_on_feature_value = kwargs.get('split_on_feature_value', '2024-03-15')
    target = kwargs.get('target', 'duration')

    df = clean(df)

    # For homework Q.3
    print(df.shape[0])

    # df = combine_features(df)
    df_train = select_features(df, features=[target]) #, features=[split_on_feature, target])

    # df_train, df_val = split_on_value(
    #     df,
    #     split_on_feature,
    #     split_on_feature_value,
    # )

    df_val = select_features(clean(pd.read_parquet("/home/src/taxi_data/yellow_tripdata_2023-04.parquet"), False), features=[target])


    return df_train, df_val