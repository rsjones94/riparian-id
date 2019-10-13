import os
os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'

import random
from copy import copy
from shutil import copyfile

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution, brute
import laspy

from optimization_helpers import *
from filter_rasters import *
from preprocessing_tools import *
from rasteration import *

# '180500020905'
# ARRA-CA_GoldenGate_2010_000878.las needed to be reexported to las

refs = { # EPSG codes
        '010500021301': 26919,
        '030902040303': 2777,
        '070801050901': 26915,
        '080102040304': 3723,
        '130202090102': 26913,
        '140801040103': 26913,
        '180500020905': 26910
        }

par = r'E:\gen_model\study_areas'
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
    cut_target = os.path.join(working,'study_LiDAR','products','clipped')
    copy_tiles(rasteration_target, copy_target)

    cut_shape = os.path.join(working,'study_area','study_area_r.shp')

    ref_code = refs[sub]
    mosaic_folders(copy_target, cut_target, cut_shape, ref_code)