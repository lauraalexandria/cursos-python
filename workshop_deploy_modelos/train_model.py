# Ativando ambiente virtual no terminal: C:\Users\laura\programaria\Scripts\activate

import os  # Evitar erros de leitura e caminho;
import sys
sys.path.insert(1, os.getcwd())

import pandas as pd
from feature_engineering.feature_engineering_pipeline import FeatureEngineering # Importando o pipeline da pasta
from sklearn.linear_model import LogisticRegression
from src.utils import save_pickle 
from sklearn.metrics import accuracy_score

def nain():
    train = pd.read_csv("data/train.csv", index_col = "PassengerId")
    X_train = train.drop(["Survived"], axis = 1)
    Y_train = train["Survived"]
     
    test = pd.read_csv("data/test.csv", index_col = "PassengerId")
     
    features = ["Pclass", "Age", "Sex"]
    X_train = X_train[features]
    Y_train = Y_train[features]
     
    feature_engineering_pipeline = FeatureEngineering(numerical_features=['Pclass', 'Age']).get_pipeline()
    index = X_train.index
    X_train = feature_engineering_pipeline.fit_transform(X_train) # Aplica o tratamento.
    X_test['PassengerId'] = index
    X_test = X_test.set_index('PassengerId')
    
    index = X_test.index
    test = feature_engineering_pipeline.fit_transform(test) # Aplica o tratamento.
    test['PassengerId'] = index
    test = test.set_index('PassengerId')
    
    X_train.to_csv('data/train_after_feature_engineering.csv')
    test.to_csv('data/test_after_feature_engineering.csv')
    
    model = LogisticRegression(verbose=1, max_iter=1000)
    model.fit(X_train, y_train)
    save_pickle(model, 'models/model.pkl')
    
    X_train['prediction'] = model.predict_proba(X_train)[:,1] # Probabilidade de não sobreviver é a  coluna 0;
    X_train['Survived'] = model.predict(X_train.drop('prediction', axis = 1))
    print(f'Acurácia do conjunto de treinamento: {accuracy_score(y_train, df["Survived"])}')
     
    test.to_csv('data/test_predictions.csv')
    
    
if __name__ == '__main__':
    main()