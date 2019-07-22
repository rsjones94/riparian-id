import os

import rasterio as rio
import os
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
import matplotlib as mpl

from osgeo import gdal

parent_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
sub_folders = os.listdir(parent_folder)

sub_folders = [sub_folders[2]]

extensions = {'all_intensity': '_allintens.tif',
              'dem': '_dem.tif',
              'dem_slope': '_demslope.tif',
              'dhm': '_dhm.tif',
              'dsm': '_dsm.tif',
              'dsm_slope': '_dsmslope.tif',
              'first_intensity': '_firstintens.tif',
              'first_intensity_slope': '_firstintensslope.tif',
              'n_returns': '_nreturns.tif'
              }

output_folder = 'trees_prediction'

for folder in sub_folders:
    files = {key: os.path.join(parent_folder, folder, folder+ext) for key, ext in extensions.items()}

    data = {}
    for key, file in files.items():
        with rio.open(os.path.join(file), 'r') as src:
            data[key] = src.read(1)
            old_meta = src.profile

    work_folder = os.path.join(parent_folder, folder, output_folder)
    os.mkdir(work_folder)
    
    dhm_mask = data['dhm'] >= 3
    n_returns_mask = data['n_returns'] > 1.5

    sum_mask = np.logical_and(dhm_mask, n_returns_mask)
    final_mask = sum_mask.astype(int)

    out_data = sum_mask.astype(int)

    old_meta['dtype'] = str(out_data.dtype)
    with rio.open(os.path.join(parent_folder, folder, output_folder, 'trees.tif'), 'w', **old_meta) as dst:
        dst.write(out_data, 1)
