import numpy as np


def remove_nan(vector):
    vector[np.isnan(vector)] =  np.nanmean(vector)
    return vector
