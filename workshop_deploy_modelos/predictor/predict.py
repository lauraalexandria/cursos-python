from feature_engineering.feature_engineering_pipeline import FeatureEngineering

import settings as st

class Predict:
    
    def __init__(self):                  # Importando modelo;
        self.model_path = st.model_path
        self.model = load_pickle(self.model_path)

    def predict(self, data):
        processed_data = FeatureEngineering(st.numerical_features).get_pipeline().transform(data)
        result = self.model.predict(processed_data)
        return result
