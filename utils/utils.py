import numpy as np
import random

import config

def remove_nan(vector):
    vector[np.isnan(vector)] =  np.nanmean(vector)
    return vector

def shuffle(l1, l2):
    combined_data = list(zip(l1, l2))
    random.Random(config.SEED).shuffle(combined_data)
    l1, l2 = zip(*combined_data)
    return l1, l2


