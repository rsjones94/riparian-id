import random
import os
from copy import copy
from shutil import copyfile

from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution, brute
import laspy

from optimization_helpers import *
from filter_rasters import *
from preprocessing_tools import *
from rasteration import *

# '180500020905'
# ARRA-CA_GoldenGate_2010_000878.las
# lasoptimize -i "D:\SkyJones\gen_model\study_areas\180500020905\LiDAR\2010_golden\las\ARRA-CA_GoldenGate_2010_000878.las" -o "C:\Users\rj3h\Desktop\test.las"
# file = r'D:\SkyJones\gen_model\study_areas\180500020905\LiDAR\2010_golden\las\ARRA-CA_GoldenGate_2010_000878.las'


# workup
par = r'D:\SkyJones\gen_model\study_areas'
folders = os.listdir(par)
total_n = len(folders)
for i,sub in enumerate(folders):
    print(f'\n\n!!!!!!!!!!!!!!!\n Working on {sub}, {i+1} of {total_n} \n!!!!!!!!!!!!!!!\n\n')
    working = os.path.join(par,sub)
    lidar_folder = os.path.join(working,'LiDAR')

    possible = os.listdir(lidar_folder)
    year_folder = [i for i in possible if '20' in i]
    assert len(year_folder) == 1
    year_folder = year_folder[0]
    data = os.path.join(lidar_folder,year_folder,'las')
    rasteration_target = os.path.join(working,'study_LiDAR','products','tiled')
    rasteration(data, rasteration_target, resolution=1)
    copy_target = os.path.join(working,'study_LiDAR','products','mosaic')
    copy_tiles(rasteration_target, copy_target)

    mosaic_folders(copy_target)


"""
# move nreturns files
par = r'D:\SkyJones\gen_veg\sys2_analysis\lidar\2012_tn\system2_overlap\nret'
big_dest = r'D:\SkyJones\gen_veg\sys2_analysis\lidar\2012_tn\system2_overlap\las_products'
files = os.listdir(par)
for i,file in enumerate(files):
    print(i,file)
    src = os.path.join(par,file)
    dst = os.path.join(big_dest,file[:-13],file)
    # os.remove(dst)
    copyfile(src, dst)
"""

"""
# generate new _nreturn.las tiles
par = r'D:\SkyJones\gen_veg\sys2_analysis\lidar\2012_tn\system2_overlap\overlap_las'
new = r'D:\SkyJones\gen_veg\sys2_analysis\lidar\2012_tn\system2_overlap\las_products'
files = os.listdir(par)
for i,file in enumerate(files):
    print(i,file)
    full = os.path.join(par,file)
    nreturnsname = file[:-4] + '_nreturns.tif'
    nreturns_file = os.path.join(new, nreturnsname)
    wbt.lidar_tin_gridding(i=full,
                           output=nreturns_file,
                           parameter='user data',
                           returns='last',
                           resolution=1,
                           exclude_cls='7,13,14,18')
"""

"""
# use the user_data field to store return number data
par = r'D:\SkyJones\gen_veg\sys2_analysis\lidar\2012_tn\system2_overlap\overlap_las'
files = os.listdir(par)
for i,file in enumerate(files):
    full = os.path.join(par,file)
    print(i,file)
    in_file = laspy.file.File(full, mode="rw")
    in_file.user_data = copy(in_file.return_num)
    in_file.close()
"""

"""
### sample output
random.seed(20)

filter1 = [filter_greater, 1.2, 0]
filter2 = [filter_greater, 3, 0]
filter3 = [filter_greater, 1.2, 0, filter_greater, 3, 0]

extension1 = ['_nreturns.tif']
extension2 = ['_dhm.tif']
extension3 = ['_nreturns.tif','_dhm.tif']

out1 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\example_filters\nreturn_1p2'
out2 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\example_filters\dhm_3'
out3 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\example_filters\combined'

filters = [filter1, filter2, filter3]
extensions = [extension1, extension2, extension3]
outs = [out1, out2, out3]

n = 10
parent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
all_subs = os.listdir(parent)
sub_folders = random.sample(all_subs, n)
for fil, ext, o in zip(filters, extensions, outs):
    demonstrate_filter(fil, ext, parent, sub_folders, o)
"""

### optimization
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