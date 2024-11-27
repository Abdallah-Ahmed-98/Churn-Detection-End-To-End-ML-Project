from mlProject.config.configuration import ConfigurationManager
from mlProject.components.prediction import Prediction


STAGE_NAME = "Prediction stage"

class PredictionPipeline:
    def __init__(self):
        pass

    def main(self, data):
        config = ConfigurationManager()
        prediction_config = config.get_prediction_config()
        prediction = Prediction(config=prediction_config)
        return prediction.predict(data)