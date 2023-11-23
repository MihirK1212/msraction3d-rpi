import numpy as np
import joblib
import os

from utils.model_base import EnsembleMemberModel

class KNNClassifierModel(EnsembleMemberModel):

    def __init__(self) -> None:
        self.loaded_trainer = False
        
    def get_predictions(self, X):
        predictions = self.model.predict(X)
        predictions = np.array(predictions, dtype=np.int64)
        return predictions

    def load_model_state(self, X, dir, model_name):
        if not self.loaded_trainer:
            print('Loading trainer', model_name)
            self.model = joblib.load(os.path.join(dir, model_name+'.joblib'))
            self.loaded_trainer = True
