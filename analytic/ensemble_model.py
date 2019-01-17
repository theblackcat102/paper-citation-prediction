from sklearn.base import TransformerMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier

class RidgeTransformer(Ridge, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X).reshape(-1, 1)


class RandomForestTransformer(RandomForestRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X).reshape(-1, 1)


class KNeighborsTransformer(KNeighborsRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X).reshape(-1, 1)

class GradientBoostTransformer(GradientBoostingRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X).reshape(-1, 1)


class XgboostTransformer(XGBRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X).reshape(-1, 1)

class CatboostTransformer(CatBoostRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X).reshape(-1, 1)

class LinearRegressionTransformer(LinearRegression, TransformerMixin):
    def transform(self, X, *_):
        return self.predict(X).reshape(-1, 1)
    
def build_model():
    catboost1_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly_feats', PolynomialFeatures()),
        ('ridge', RidgeTransformer()),
        ('catboost', CatboostTransformer(verbose=0)),
    ])


    ridge_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        # ('poly_feats', PolynomialFeatures()),
        # ('ridge', RidgeTransformer()),

    ])

    pred_union = FeatureUnion(
        transformer_list=[
            # ('linear', LinearRegressionTransformer()),
            # ('catboost1_transformer', catboost1_transformer),
            ('xgboost', XgboostTransformer(colsample_bytree=0.1,
                max_depth=4,
                verbose=0)),
            ('catboost2', CatboostTransformer(n_estimators=10,verbose=0)),
            ('xgboost2', XgboostTransformer(verbose=0)),
            ('catboost', CatboostTransformer(verbose=0)),
            ('gradboost', GradientBoostTransformer())
        ],
        n_jobs=5
    )

    model = Pipeline(steps=[
        ('pred_union', pred_union),
        ('xgboost', CatBoostRegressor( verbose=0))
    ])

    return model
