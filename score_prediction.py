import os

import numpy as np
import rasterio as rio
import pandas as pd

pred = r'D:\SkyJones\lidar\2012_tn\system2_overlap\16sbe9493\FILTERED.tif'
val = r'D:\SkyJones\lidar\2012_tn\system2_overlap\16sbe9493\16sbe9493_naipcover.tif'

def score_prediction(prediction, validator):
    """
    Outputs a score from 0 to 1 that quantifies how close a binary raster (the prediction) matches another
    binary raster (the validator). The rasters must have the same extent and resolution.

    Args:
        prediction: a binary raster with 1 band
        validator: a binary raster with 1 band

    Returns: a tuple where the first entry is a float from -1 to 1, where 1 is a perfect match and a -1 is
             a perfect inversion, and the second entry is a confusion matrix
    """
    with rio.open(os.path.join(prediction), 'r') as src:
        prediction_data = src.read(1)
        metadata = src.profile
    with rio.open(os.path.join(validator), 'r') as src:
        validator_data = src.read(1)

    true_positives = np.logical_and(prediction_data == 1, validator_data == 1).astype(int)
    true_negatives = np.logical_and(prediction_data == 0, validator_data == 0).astype(int)
    false_positives = np.logical_and(prediction_data == 1, validator_data == 0).astype(int)
    false_negatives = np.logical_and(prediction_data == 0, validator_data == 1).astype(int)

    n_tp = sum(sum(true_positives))
    n_tn = sum(sum(true_negatives))
    n_fp = sum(sum(false_positives))
    n_fn = sum(sum(false_negatives))

    counts = [n_tp, n_tn, n_fp, n_fn]
    n_tot = sum(counts)
    fracs = [c/n_tot for c in counts]

    confusion = {'tp': fracs[0],
                 'tn': fracs[1],
                 'fp': fracs[2],
                 'fn': fracs[3]}
    score = confusion['tp'] + confusion['tn'] - confusion['fp'] - confusion['fn']

    return score, confusion


s,c = score_prediction(pred, val)
