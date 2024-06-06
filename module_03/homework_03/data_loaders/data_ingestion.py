import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

MONTHS = [3]

@data_loader
def load_data(*args, **kwargs):

    dfs = []
    for mon in MONTHS:
        dfs.append(pd.read_parquet(f"/home/src/taxi_data/yellow_tripdata_2023-{mon:02d}.parquet"))

    return pd.concat(dfs)
