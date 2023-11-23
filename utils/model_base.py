from abc import ABC, abstractmethod

class EnsembleMemberModel(ABC):

    @abstractmethod
    def get_predictions(self, X):
      pass
    
    @abstractmethod
    def load_model_state(self, X, dir, model_name):
      pass