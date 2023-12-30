import numpy as np
from collections import Counter
import json

import constants
import data.parse as data_parse

from models.neural_network import NeuralNetworkClassifierModel
from models.random_forest import RandomForestClassifierModel
from models.knn import KNNClassifierModel
from models.svm import SVMClassifierModel


class MultimethodEnsemble:
    def __init__(self, training=True):
        self.training = training
        self.ensemble_trainers = self.ensemble_trainers = [
            {
                "model": NeuralNetworkClassifierModel(lr_decay_epochs=100),
                "aggregation_method": constants.STATISTICAL_MOMENTS,
                "name": "nn1_statistical_moments",
            },
            {
                "model": NeuralNetworkClassifierModel(
                    lr=0.01, lr_decay_epochs=100, num_epochs=200
                ),
                "aggregation_method": constants.STATISTICAL_MOMENTS,
                "name": "nn2_statistical_moments",
            },
            {
                "model": NeuralNetworkClassifierModel(
                    lr=0.1,
                    lr_decay_epochs=200,
                    num_epochs=200,
                    num_hidden_layers=2,
                    hidden_layer_dim_factor=[1, 1],
                ),
                "aggregation_method": constants.STATISTICAL_MOMENTS,
                "name": "nn3_statistical_moments",
            },
            {
                "model": NeuralNetworkClassifierModel(
                    lr=0.1, lr_decay_epochs=100, num_epochs=200
                ),
                "aggregation_method": constants.STATISTICAL_MOMENTS,
                "name": "nn4_statistical_moments",
            },
            {
                "model": NeuralNetworkClassifierModel(
                    lr=0.1, lr_decay_end_factor=1.0, lr_decay_epochs=100, num_epochs=100
                ),
                "aggregation_method": constants.STATISTICAL_MOMENTS,
                "name": "nn5_statistical_moments",
            },
            {
                "model": RandomForestClassifierModel(),
                "aggregation_method": constants.STATISTICAL_MOMENTS,
                "name": "rf_statistical_moments",
            },
            {
                "model": KNNClassifierModel(),
                "aggregation_method": constants.STATISTICAL_MOMENTS,
                "name": "knn_statistical_moments",
            },
            {
                "model": SVMClassifierModel(),
                "aggregation_method": constants.STATISTICAL_MOMENTS,
                "name": "svm_statistical_moments",
            },
            {
                "model": NeuralNetworkClassifierModel(lr_decay_epochs=100),
                "aggregation_method": constants.WINDOW_DIVISION,
                "name": "nn1_window_division",
            },
            {
                "model": NeuralNetworkClassifierModel(
                    lr=0.01, lr_decay_epochs=100, num_epochs=200
                ),
                "aggregation_method": constants.WINDOW_DIVISION,
                "name": "nn2_window_division",
            },
            {
                "model": NeuralNetworkClassifierModel(
                    lr=0.1,
                    lr_decay_epochs=200,
                    num_epochs=200,
                    num_hidden_layers=2,
                    hidden_layer_dim_factor=[1, 1],
                ),
                "aggregation_method": constants.WINDOW_DIVISION,
                "name": "nn3_window_division",
            },
            {
                "model": NeuralNetworkClassifierModel(
                    lr=0.1, lr_decay_epochs=100, num_epochs=200
                ),
                "aggregation_method": constants.WINDOW_DIVISION,
                "name": "nn4_window_division",
            },
            {
                "model": NeuralNetworkClassifierModel(
                    lr=0.1, lr_decay_end_factor=1.0, lr_decay_epochs=100, num_epochs=100
                ),
                "aggregation_method": constants.WINDOW_DIVISION,
                "name": "nn5_window_division",
            },
            {
                "model": RandomForestClassifierModel(),
                "aggregation_method": constants.WINDOW_DIVISION,
                "name": "rf_window_division",
            },
            {
                "model": SVMClassifierModel(),
                "aggregation_method": constants.WINDOW_DIVISION,
                "name": "svm_window_division",
            },
        ]
        self.max_accuracy_trainer_index = 0

    
    def load_training_config(self):
        with open(constants.TRAINING_CONFIG_PATH, 'r') as json_file:
            training_config = json.load(json_file)
        self.max_accuracy_trainer_index = training_config['MAX_ACCURACY_TRAINER_INDEX']

    def get_predictions(self, data):
        assert self.training == False
        self.statistical_moments_data = data_parse.get_parsed_testing_data(data, aggregation_method=constants.STATISTICAL_MOMENTS)
        self.window_division_data = data_parse.get_parsed_testing_data(data, aggregation_method=constants.WINDOW_DIVISION)
        model_predictions_list = []
        for trainer in self.ensemble_trainers:
            X = self.get_model_data(trainer["aggregation_method"])
            trainer["model"].load_model_state(X, constants.TRAINER_SAVE_DIR, trainer["name"])
            model_predictions = trainer["model"].get_predictions(X)
            model_predictions_list.append(model_predictions)
        predictions = self.get_polled_predictions(model_predictions_list)
        return predictions


    def get_model_data(self, aggregation_method):
        data_mapping = {
            constants.STATISTICAL_MOMENTS: self.statistical_moments_data,
            constants.WINDOW_DIVISION: self.window_division_data,
        }
        return data_mapping.get(aggregation_method, None)
        
    def get_majority_vote(self, votes):
        return (
            max(Counter(votes).items(), key=lambda x: x[1])[0]
            if Counter(votes).most_common(1)[0][1] > len(votes) // 2
            else votes[self.max_accuracy_trainer_index]
        )


    def get_polled_predictions(self, model_predictions_list):
        sample_wise_model_predictions = np.transpose(np.array(model_predictions_list, dtype=np.int64))
        assert sample_wise_model_predictions.shape[1] == len(self.ensemble_trainers)
        predictions = [
            self.get_majority_vote(
                row.tolist()
            ) for row in sample_wise_model_predictions
        ]
        return np.array(predictions, dtype=np.int64)
    

    
