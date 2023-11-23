import numpy as np
import os

from utils.model_base import EnsembleMemberModel
import config

if config.IS_RPI_DEVICE:
    import tflite_runtime.interpreter as tflite
else:
    import tensorflow.lite as tflite

np.random.seed(config.SEED)


class NeuralNetworkClassifierModel(EnsembleMemberModel):
    def __init__(
        self,
        lr=0.1,
        lr_decay_end_factor=0.1,
        lr_decay_epochs=50,
        num_epochs=100,
        num_hidden_layers=1,
        hidden_layer_dim_factor=[1],
    ):
        self.lr = lr
        self.lr_decay_end_factor = lr_decay_end_factor
        self.lr_decay_epochs = lr_decay_epochs
        self.num_epochs = num_epochs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_dim_factor = hidden_layer_dim_factor
        self.loaded_trainer = False

        
    def get_predictions(self, X):
        self.output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = np.argmax(self.output_data, axis=1)
        predictions = np.array(predictions, dtype=np.int64)
        return predictions

    def load_model_state(self, X, dir, model_name): 
        np.random.seed(config.SEED)
        if not self.loaded_trainer:
            print('Loading trainer', model_name)
            self.interpreter = tflite.Interpreter(os.path.join(dir, model_name+'.tflite'))
            self.loaded_trainer = True
        self.interpreter.resize_tensor_input(self.interpreter.get_input_details()[0]['index'], [X.shape[0], X.shape[1]])
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))
        self.interpreter.invoke()
