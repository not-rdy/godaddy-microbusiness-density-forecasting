import os
import sys
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from settings.base import PATH_PROJ, PATH_DATA_INTERIM
from lib.utils import save_f
sys.path.append(PATH_PROJ)
from models.catboost.params import params  # noqa: E402

if __name__ == '__main__':

    mlflow.set_experiment('catboost')
    mlflow.start_run()
    mlflow.log_params(params)

    PATH_MODEL = os.path.join(PATH_PROJ, 'models', 'catboost')

    train_catboost = pd.read_csv(
        os.path.join(PATH_DATA_INTERIM, 'train_catboost.csv'),
        index_col=0).iloc[:100]
    test_catboost = pd.read_csv(
        os.path.join(PATH_DATA_INTERIM, 'test_catboost.csv'),
        index_col=0).iloc[:100]

    x_train = train_catboost.drop('microbusiness_density', axis=1).copy()
    y_train = train_catboost['microbusiness_density'].copy()
    x_test = test_catboost.drop('microbusiness_density', axis=1).copy()
    y_test = test_catboost['microbusiness_density'].copy()
    del train_catboost, test_catboost

    model = CatBoostRegressor(**params)
    model.fit(
        cat_features=['code_state', 'code_county'],
        X=x_train,
        y=y_train,
        eval_set=(x_test, y_test))
    best_scores = model.get_best_score()

    feature_importance = pd.Series(
        data=model.feature_importances_,
        index=x_train.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    feature_importance.plot(kind='barh', grid=True, ax=ax)
    fig.savefig(
        os.path.join(PATH_MODEL, 'feature_importance.png'),
        bbox_inches='tight')

    save_f(
        filename=os.path.join(PATH_MODEL, 'model.pkl'),
        obj=model)

    mlflow.log_metric(
        key='train_SMAPE',
        value=best_scores.get('learn').get('SMAPE'))
    mlflow.log_metric(
        key='train_RMSE',
        value=best_scores.get('learn').get('RMSE'))
    mlflow.log_metric(
        key='test_SMAPE',
        value=best_scores.get('validation').get('SMAPE'))
    mlflow.log_metric(
        key='test_RMSE',
        value=best_scores.get('validation').get('RMSE'))

    mlflow.log_artifacts(PATH_MODEL)

    mlflow.end_run()
