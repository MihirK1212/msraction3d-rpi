import xgboost as xgb
import numpy as np
import os

from utils.model_base import EnsembleMemberModel
import config

if not config.IS_RPI_DEVICE:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

class XGBClassifierModel(EnsembleMemberModel):

    def __init__(self) -> None:
        self.loaded_trainer = False

    def get_predictions(self, X):
        dX = xgb.DMatrix(X)
        predictions = np.rint(self.model.predict(dX))
        predictions = np.array(predictions, dtype=np.int64)
        return predictions
    
    def load_model_state(self, X, dir, model_name):
        if not self.loaded_trainer:
            print('Loading trainer', model_name)
            self.model = xgb.Booster()
            self.model.load_model(os.path.join(dir, model_name+'.model'))
            self.loaded_trainer = True
        
    