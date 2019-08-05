import os

import numpy as np
import rasterio as rio


def score_prediction(prediction, validator):
    """
    Outputs a score from 0 to 1 that quantifies how close a binary raster (the prediction) matches another
    binary raster (the validator). The rasters must have the same extent and resolution.

    Args:
        prediction: a binary raster file with 1 band or a raw numpy array representing such a file
        validator: a binary raster with 1 band or a raw numpy array representing such a file

    Returns: a tuple where the first entry is a float from -1 to 1, where -1 is a perfect match and a 1 is
             a perfect inversion, and the second entry is a confusion matrix
    """
    if isinstance(prediction, str):
        with rio.open(os.path.join(prediction), 'r') as src:
            prediction_data = src.read(1)
    else:
        prediction_data = prediction

    if isinstance(validator, str):
        with rio.open(os.path.join(validator), 'r') as src:
            validator_data = src.read(1)
    else:
        validator_data = validator

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
    score = -(confusion['tp'] + confusion['tn'] - confusion['fp'] - confusion['fn'])

    return score, confusion

def score_predictions(predictions, validators):
    """
    Scores lists of predictions and validators. All rasters should have identical extents and resolutions.

    Args:
        predictions: a list of binary rasters with 1 band or raw numpy arrays representing such files
        validators: a corresponding list of binary rasters with 1 band or raw numpy arrays representing such files

    Returns: An overall score, and a list of tuples where in each tuple the first entry is a float from -1 to 1,
             where 1 is a perfect match and a -1 is a perfect inversion, and the second entry is a confusion matrix.
             Each tuple corresponds to a prediction-validator pair.
    """

    scores_tups = [score_prediction(prediction, validator) for
                   prediction, validator in zip(predictions, validators)]

    overall = np.mean([tup[0] for tup in score_prediction()])

    return overall, scores_tups
