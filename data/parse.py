import numpy as np
import copy

import constants
import data.statistical_moments_aggregation as statistical_moments_aggregation
import data.window_division_aggregation as window_division_aggregation


def get_mean_feature_vector(sequence, aggregation_method):
    
    agg_sequence = copy.deepcopy(sequence)

    if aggregation_method == constants.STATISTICAL_MOMENTS:
        return statistical_moments_aggregation.get_statistical_moments_aggregation(
            sequence=agg_sequence
        )

    elif aggregation_method == constants.WINDOW_DIVISION:
        return window_division_aggregation.get_window_division_aggregation(
            sequence=agg_sequence
        )

    else:
        raise ValueError


def get_parsed_testing_data(data, aggregation_method):
    X = []
    for sequence in data:
        f = get_mean_feature_vector(
            sequence=sequence, aggregation_method=aggregation_method
        )
        X.append(f)
    X = np.array(X, dtype=np.float64)
    return X
