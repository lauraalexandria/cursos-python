from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from feature_engineering.feature_engineering_pipeline import FeatureEngineering # chama o método
from feature_engineering.missing_imputer import MissingValuesImputer
from feature_engineering.numerical_scaler import NumericalFeaturesScaler

class FeatureEngeneering(BaseEstimator, TransformerMixin):
    def get_pipeline(self):
        return Pipeline(
            [
            ("ohe", OneHotEncode()), # Primeira transformação: one-hot enconding;
            ("missing_imputer", MissingValuesImputer()),
            ("numerical_scaler", NumericalFeaturesScaler(numerical_features=self.numerical_features)),
            ]
            
            )
    

