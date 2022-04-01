from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.utils import load_pickle, save_pickle

class OneHotEncode(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature = 'Sex'
        
        
    def fit (self, df):
        df[self.feature] = df[self.feature].astype(str)
        
        enc = OneHotEnconder(handle_unknown = 'ignore', drop = 'if_binary')
        # Vai ignorar novas categorias se aparecerem e cria 1 variável por ser binário;
        
        enc.fit(df[self.feature])
        
        save_pickle(enc, self.pickle_path)

        return self
    
    def tranformer(self, df):
        df[self.feature] = df[self.feature].astype(str)
        
        enc = load_pickle(self.pickle_path)
        
        df = self.append_ohe_to_dataframe(enc, df)
        
        df = df.drop(columns=self.feature)
        
        return df
        
    def append_ohe_to_dataframe(self, enc, df):     
        ohe_feature = enc.transform(df[self.feature].values.reshape(-1, 1)).toarray()
        df_ohe = pd.DataFrame(ohe_feature, columns=enc.get_feature_names_out([self.feature]))
        df_ohe.index = df.index
        
        return pd.concat([df, df_ohe], axis=1)
