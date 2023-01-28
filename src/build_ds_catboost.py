import os
import pandas as pd
from argparse import ArgumentParser
from settings.base import PATH_DATA_RAW, PATH_DATA_INTERIM
from lib.dataset_builder import DSBuilder


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '-jobs',
        type=int,
        help='n processes')
    parser.add_argument(
        '-lag',
        type=int,
        help='lag for merge')

    args = parser.parse_args()
    n_jobs = args.jobs
    lag = args.lag

    train = pd.read_csv(
        os.path.join(PATH_DATA_RAW, 'train.csv'))
    census_starter = pd.read_csv(
        os.path.join(PATH_DATA_RAW, 'census_starter.csv'))

    builder = DSBuilder()
    train, test = builder.build(
        train=train,
        census_starter=census_starter,
        merged_lag=lag,
        model_type='catboost',
        n_jobs=n_jobs)

    train.to_csv(
        os.path.join(PATH_DATA_INTERIM, 'train_catboost.csv'))
    test.to_csv(
        os.path.join(PATH_DATA_INTERIM, 'test_catboost.csv'))
