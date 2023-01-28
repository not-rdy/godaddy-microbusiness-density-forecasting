import pandas as pd
from prophet import Prophet
from tqdm import tqdm
from typing import Tuple
from multiprocessing import Pool
from .utils import suppress_stdout_stderr


class DSBuilder:

    def __init__(self) -> None:
        self.available_model_types = ['catboost', 'GCN']

    @staticmethod
    def __preproc(
            train: pd.DataFrame,
            census_starter: pd.DataFrame,
            merged_lag: int) -> Tuple[pd.DataFrame]:
        train['first_day_of_month'] = pd.to_datetime(
            train['first_day_of_month'], format='%Y-%m-%d')
        train['year_with_lag'] = train['first_day_of_month'].dt.year -\
            merged_lag
        train['cfips'] = train['cfips'].astype(str)
        census_starter['cfips'] = census_starter['cfips'].astype(str)
        census_starter = census_starter.set_index('cfips')
        return train, census_starter

    @staticmethod
    def __merge(
            train: pd.DataFrame,
            census_starter: pd.DataFrame) -> pd.DataFrame:
        colLong_colsWide = {
            'pct_bb': [x for x in census_starter.columns if 'pct_bb' in x],
            'pct_college': [
                x for x in census_starter.columns if 'pct_college' in x],
            'pct_foreign_born': [
                x for x in census_starter.columns if 'pct_foreign_born' in x],
            'pct_it_workers': [
                x for x in census_starter.columns if 'pct_it_workers' in x],
            'median_hh_inc': [
                x for x in census_starter.columns if 'median_hh_inc' in x]
        }
        for long_name, wide_names in colLong_colsWide.items():
            part_to_long = pd.melt(
                census_starter,
                value_vars=wide_names,
                value_name=long_name,
                ignore_index=False)
            part_to_long['year'] = part_to_long['variable']\
                .map(lambda x: int(x.split('_')[-1]))
            part_to_long = part_to_long.drop('variable', axis=1)
            part_to_long = part_to_long.reset_index()
            train = train.merge(
                right=part_to_long,
                how='left',
                left_on=['cfips', 'year_with_lag'],
                right_on=['cfips', 'year'])
            train = train.drop('year', axis=1)
        train = train.drop('year_with_lag', axis=1)
        return train

    @staticmethod
    def _get_prophet_features(df: pd.DataFrame) -> pd.DataFrame:

        dates_all = pd.Series(df['first_day_of_month'].unique())\
            .sort_values(ascending=True)
        dates_train = dates_all[dates_all <= '2021-10-01'].tolist()
        dates_test = dates_all[dates_all > '2021-10-01'].tolist()

        df = df.set_index('first_day_of_month')
        ts1 = df.loc[dates_train, 'microbusiness_density']
        ts2 = df.loc[dates_train, 'active']
        ts1 = ts1.reset_index('first_day_of_month')
        ts2 = ts2.reset_index('first_day_of_month')
        ts1 = ts1.sort_values(by='first_day_of_month', ascending=True)
        ts2 = ts2.sort_values(by='first_day_of_month', ascending=True)
        colrename = {
            'first_day_of_month': 'ds',
            'microbusiness_density': 'y',
            'active': 'y'}
        ts1 = ts1.rename(columns=colrename)
        ts2 = ts2.rename(columns=colrename)
        with suppress_stdout_stderr():
            prophet1 = Prophet()
            prophet2 = Prophet()
            prophet1.fit(ts1)
            prophet2.fit(ts2)
            future1 = prophet1\
                .make_future_dataframe(periods=len(dates_test), freq='MS')
            future2 = prophet2\
                .make_future_dataframe(periods=len(dates_test), freq='MS')
            forecast1 = prophet1.predict(future1).set_index('ds')
            forecast2 = prophet2.predict(future2).set_index('ds')
        cols_drop = [
            'multiplicative_terms',
            'multiplicative_terms_lower',
            'multiplicative_terms_upper']
        features1 = forecast1.drop(cols_drop, axis=1)
        features2 = forecast2.drop(cols_drop, axis=1)
        colnames1 = [
            'microbusiness_density_' + x for x in features1.columns]
        colnames2 = [
            'active_' + x for x in features2.columns]
        features1.columns = colnames1
        features2.columns = colnames2
        for col1, col2 in zip(colnames1, colnames2):
            df[col1] = None
            df[col2] = None
        df.loc[features1.index, colnames1] = features1.values
        df.loc[features2.index, colnames2] = features2.values
        df_train = df.loc[dates_train, :].reset_index()
        df_test = df.loc[dates_test, :].reset_index()
        return df_train, df_test

    def __build_catboost(self, ds: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
        ds['code_state'] = ds['cfips'].map(lambda x: x[:2])
        ds['code_county'] = ds['cfips'].map(lambda x: x[2:])
        ds['day'] = ds['first_day_of_month'].dt.day
        ds['month'] = ds['first_day_of_month'].dt.month
        ds['year'] = ds['first_day_of_month'].dt.year

        print('split ds to multiproc ...')
        dfs_list = []
        for cfip in tqdm(ds['cfips'].unique(), total=ds['cfips'].nunique()):
            df = ds[ds['cfips'] == cfip].copy()
            dfs_list.append(df)
        del ds

        print('start prepare ...')
        tuples_list = []
        with Pool(n_jobs) as p:
            for res in tqdm(
                    p.imap_unordered(self._get_prophet_features, dfs_list),
                    total=len(dfs_list)):
                tuples_list.append(res)
        train_list = [tup[0] for tup in tuples_list]
        test_list = [tup[1] for tup in tuples_list]

        train = pd.concat(train_list).reset_index(drop=True)
        test = pd.concat(test_list).reset_index(drop=True)
        cols_drop = [
            'row_id', 'county', 'state',
            'active', 'cfips', 'first_day_of_month']
        train = train.drop(cols_drop, axis=1)
        test = test.drop(cols_drop, axis=1)
        return train, test

    def build(
            self, train: pd.DataFrame,
            census_starter: pd.DataFrame,
            merged_lag: int,
            model_type: str,
            n_jobs: int) -> Tuple[pd.DataFrame]:
        f"""
        model_type: {self.available_model_types}
        """

        if model_type not in self.available_model_types:
            raise TypeError(
                f"available model types: {self.available_model_types}")

        train, census_starter = self.__preproc(
            train, census_starter, merged_lag)
        ds = self.__merge(train, census_starter)
        del train, census_starter

        if model_type == 'catboost':
            train, test = self.__build_catboost(ds, n_jobs)

        return train, test
