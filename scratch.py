import random

from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution, brute

from optimization_helpers import *
from filter_rasters import *

"""
random.seed(20)

filter1 = [filter_greater, 1.2, 0]
extensions = ['_nreturns.tif']
parent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
all_subs = os.listdir(parent)
n = 1
out_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\example_filters'

sub_folders = random.sample(all_subs, n)
demonstrate_filter(filter1, extensions, parent, sub_folders, out_folder)
"""

#### begin dif ev optimization
cat_bounds = (0.51, 2.49) # only allow high and low pass filters
# input
parent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
n_tiles = 20
pop_mult = 3
max_n_iter = 50
extensions = ['_dhm.tif', '_nreturns.tif', '_dsmslope.tif', '_allintens.tif']
# should write a function to produce these bounds based on extensions
bounds = [cat_bounds, (0,50), (0,0),
          cat_bounds, (1,4), (0,0),
          cat_bounds, (0,90), (0,0),
          cat_bounds, (0,5000), (0,0)]
validator_extension = '_valtile.tif'

## end input
subfolders = os.listdir(parent)
selected_folders = random.sample(subfolders, n_tiles)
# popsize = len(bounds)*pop_mult

result = differential_evolution(multi_objective, bounds,
                                args=(extensions, parent, selected_folders, validator_extension, True),
                                maxiter=max_n_iter, popsize=pop_mult, tol=0.005, recombination=0.8, mutation=(0.7,1.5))
result_inter = interpret(extensions, result.x)

"""
#### begin brute optimization
# input
parent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
n_tiles = 50
extensions = ['_dhm.tif', '_nreturns.tif']
# should write a function to produce these bounds based on extensions
ranges = [(1,1), slice(0,5,.1), (0,0), (1,1), slice(1,2.05,.05), (0,0)]
validator_extension = '_valtile.tif'

## end input
subfolders = os.listdir(parent)
selected_folders = random.sample(subfolders, n_tiles)
# popsize = len(bounds)*pop_mult

result = brute(func=multi_objective, ranges=ranges, Ns=1,
               args=(extensions, parent, selected_folders, validator_extension, True),
               full_output=True,
               finish=None)

res = np.stack([*result[2], result[3]], -1)

plt_varies = ([0, 'vary', 0, 0, 0, 0], [0, 0, 0, 0, 'vary', 0])
for to_plt in plt_varies:
    title, ex = extracted_title(res, to_plt, extensions)
    extracted = extract_param(res, to_plt)
    plt.figure()
    plt.plot(extracted[0], extracted[1])
    plt.title(title)
    plt.xlabel(ex)
    plt.ylabel('objective')
    plt.show()
"""

"""
#### begin manual optimization
# input
parent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
n_tiles = 30
extensions = ['_dhm.tif', '_nreturns.tif']
params = [filter_greater, 0, 0, filter_greater, 1.178, 0]
validator_extension = '_valtile.tif'

## end input
subfolders = os.listdir(parent)
selected_folders = random.sample(subfolders, n_tiles)
score, subscores = multi_objective(params, extensions, parent, selected_folders, validator_extension, as_objective=False)
name_pairs = zip(selected_folders, subscores)
name_pairs = [pair for pair in name_pairs]
name_pairs.sort(key=lambda x: x[1][0]) # sorted by lowest (best) score
"""