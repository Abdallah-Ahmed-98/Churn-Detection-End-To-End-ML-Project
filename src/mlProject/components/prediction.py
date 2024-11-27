import pandas as pd
import numpy as np
from mlProject import logger
import joblib
from mlProject.entity.config_entity import PredictionConfig
from pathlib import Path

class Prediction:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = joblib.load(Path(config.model_path))
        self.transformer = joblib.load(Path(config.transformation_pipline_path))
    

    def predict(self, data):
        """
        Preprocess the input data, then predict the outcome using the trained model.
        """
        logger.info("[Prediction] Starting prediction process...")

        # Ensure the input data is a DataFrame (if it's not already)
        if isinstance(data, pd.DataFrame) is False:
            data = pd.DataFrame(data)

        ## Drop first 3 features
        data.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

        # Apply data transformation
        transformed_data = self.transformer.transform(data)

        # Make predictions using the trained model
        predictions = self.model.predict(transformed_data)
        
        logger.info(f"[Prediction] Prediction is {predictions}.")
        logger.info("[Prediction] Prediction completed.")

        return predictions[0]