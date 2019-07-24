import os, time, random

from scipy.optimize import differential_evolution

from score_prediction import score_prediction
from filter_rasters import *


def single_objective(params_list, targets, validator, as_objective=True):
    """
    Gives a score for a tree cover prediction that results from applying the filter specified by params list to the
    targets. Evaluated against the validator

    Args:
        params_list: see filter_rasters()
        targets: see filter_rasters()
        validator: see score_prediction()
        as_objective: a bool indicating if this is an objective function. If only the score is returned. Otherwise the
                      confusion matrix is returned as well

    Returns: a float from -1 to 1, where -1 indicates the prediction is a perfect match and a 1 is
             a perfect inversion

    """
    """
    print('SINGLE OBJECTIVE!')
    print(f'params_list: {params_list}')
    print(f'targets: {targets}')
    print(f'validator: {validator}')
    """
    prediction = filter_rasters(params_list, targets)
    score, con = score_prediction(prediction, validator)
    if as_objective:
        return score
    else:
        return score, con

called = 0
def multi_objective(params_list, extensions, parent_folder, subfolders, validator_ext, as_objective=True):
    """
    Gives a score for a filter applied to multiple tiles

    Args:
        params_list: see filter_rasters()
        extensions: a list of extensions (e.g., '_dhm.tif' that can be used to build filenames pointing to a desired
                    raster
        parent_folder: the directory that each tile subdirectory is in
        subfolders: a list of folder names containing the data for the tiles to evaluate the filter against
        validator_ext: an extension that can be used to build the filename of the validation data
        as_objective: a bool indicating if this is an objective function. If only the overall score is returned.
                      else, the overall score is returned as long as a list of tuples of each tile's score and confusion
                      matrix

    Returns: a float from -1 to 1, where -1 indicates the prediction is a perfect match and a 1 is
             a perfect inversion

    """

    """
    print('MULTI OBJECTIVE!')
    print(f'params_list: {params_list}')
    print(f'extensions: {extensions}')
    print(f'parent_folder: {parent_folder}')
    print(f'subfolders: {subfolders}')
    print(f'validator_ext: {validator_ext}')
    """
    scores = []
    for subfolder in subfolders:
        if not as_objective:
            print(f'Analyzing {subfolder}')
        files = [os.path.join(parent_folder, subfolder, subfolder+ext) for ext in extensions]
        scores.append(single_objective(params_list,
                                       files,
                                       os.path.join(parent_folder, subfolder, subfolder+validator_ext),
                                       as_objective))

    if as_objective: # scores is just the score of each tile
        global called
        called += 1
        return np.mean(scores)
    else: # scores is a list of tuples: the score of the tile and the confusion matrix
        avg = np.mean([score[0] for score in scores])
        return avg, scores

"""
#### begin heuristic optimization
cat_bounds = (0.51, 2.49) # only allow high and low pass filters
# input
parent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
n_tiles = 15
pop_mult = 2
max_n_iter = 10000
extensions = ['_dhm.tif', '_nreturns.tif']
# should write a function to produce these bounds based on extensions
bounds = [cat_bounds, (0,50), (0,0), cat_bounds, (1,4), (1,1)]
validator_extension = '_valtile.tif'

## end input
subfolders = os.listdir(parent)
selected_folders = random.sample(subfolders, n_tiles)
# popsize = len(bounds)*pop_mult

result = differential_evolution(multi_objective, bounds,
                                args=(extensions, parent, selected_folders, validator_extension, True),
                                maxiter=max_n_iter, popsize=pop_mult)
"""

#### begin manual optimization
# input
parent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
n_tiles = 20
extensions = ['_dhm.tif', '_nreturns.tif']
params = [filter_greater, 3, 0, filter_greater, 1.5, 0]
validator_extension = '_valtile.tif'

## end input
subfolders = os.listdir(parent)
selected_folders = random.sample(subfolders, n_tiles)
score, subscores = multi_objective(params, extensions, parent, selected_folders, validator_extension, as_objective=False)