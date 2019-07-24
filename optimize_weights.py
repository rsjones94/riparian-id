import os, time, random

from scipy.optimize import differential_evolution

from score_prediction import score_prediction
from filter_rasters import *


def single_objective(params_list, targets, validator):
    """
    Gives a score for a tree cover prediction that results from applying the filter specified by params list to the
    targets. Evaluated against the validator

    Args:
        params_list: see filter_rasters()
        targets: see filter_rasters()
        validator: see score_prediction()

    Returns: a float from -1 to 1, where 1 indicates the prediction is a perfect match and a -1 is
             a perfect inversion

    """
    # an objective function that deals with applying a filter to a single tile
    print('SINGLE OBJECTIVE!')
    print(f'params_list: {params_list}')
    print(f'targets: {targets}')
    print(f'validator: {validator}')
    prediction = filter_rasters(params_list, targets)
    score, con = score_prediction(prediction, validator)

    return score

def multi_objective(params_list, extensions, parent_folder, subfolders, validator_ext):
    """


    Args:
        params_list: see filter_rasters()
        extensions: a list of extensions (e.g., '_dhm.tif' that can be used to build filenames pointing to a desired
                    raster
        parent_folder: the directory that each tile subdirectory is in
        subfolders: a list of folder names containing the data for the tiles to evaluate the filter against
        validator_ext: an extension that can be used to build the filename of the validation data

    Returns: a float from -1 to 1, where 1 indicates the prediction is a perfect match and a -1 is
             a perfect inversion

    """
    print('MULTI OBJECTIVE!')
    print(f'params_list: {params_list}')
    print(f'extensions: {extensions}')
    print(f'parent_folder: {parent_folder}')
    print(f'subfolders: {subfolders}')
    print(f'validator_ext: {validator_ext}')
    scores = []
    for subfolder in subfolders:
        files = [os.path.join(parent_folder, subfolder, subfolder+ext) for ext in extensions]
        scores.append(single_objective(params_list, files, os.path.join(parent_folder, subfolder, subfolder+validator_ext)))

    return np.mean(scores)


#### begin model
cat_bounds = (-0.49,4.49)
# input
parent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
n_tiles = 10
extensions = ['_dhm.tif', '_nreturns.tif']
# should write a function to produce these bounds based on extensions
bounds = [cat_bounds, (0,50), (0,50), cat_bounds, (1,8), (1,8)]
validator_extension = '_valtile.tif'

## end input
subfolders = os.listdir(parent)
selected_folders = random.sample(subfolders, n_tiles)

result = differential_evolution(multi_objective, bounds, args=(extensions, parent, selected_folders, validator_extension))



"""
p = [filter_greater, 10, 0, filter_greater, 2, 0]
# system 1
dhm1 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products\16sbe9990\16sbe9990_dhm.tif'
nret1 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products\16sbe9990\16sbe9990_nreturns.tif'
val1 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products\16sbe9990\16sbe9990_valtile.tif'

# system 2
dhm2 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products\16sbe9991\16sbe9991_dhm.tif'
nret2 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products\16sbe9991\16sbe9991_nreturns.tif'
val2 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products\16sbe9991\16sbe9991_valtile.tif'

### scores
s1 = single_objective(p, [dhm1, nret1], val1)
s2 = single_objective(p, [dhm2, nret2], val2)

exts = ['_dhm.tif', '_nreturns.tif']
parent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
subfolders = ['16sbe9990', '16sbe9991']
val_ext = '_valtile.tif'

combined_score = multi_objective(p, exts, parent, subfolders, val_ext)

print(s1, s2, combined_score)
"""