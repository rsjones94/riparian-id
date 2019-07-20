import os

import rasterio as rio
import os
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
plt.ion()
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (14, 14)
mpl.rcParams['axes.titlesize'] = 20

parent_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
sub_folders = os.listdir(parent_folder)

sub_folders = [sub_folders[2]]

for folder in sub_folders:
    work_folder = os.path.join(parent_folder, folder, 'working')
    os.mkdir(work_folder)
    
    with rio.open(os.path.join(parent_folder, folder, folder+'_dhm.tif'), 'r') as src:
        data = src.read(1)
        old_meta = src.profile
    mask = data > 3
    new_data = mask.astype(int)

    write_tf = old_meta["transform"]
    write_crs = old_meta["crs"]

    """
    fig, ax = plt.subplots(figsize=(12, 6))
    nd = ax.imshow(new_data)
    ax.set(title="Filter")
    ax.set_axis_off()
    """

    old_meta['dtype'] = str(new_data.dtype)
    with rio.open(os.path.join(parent_folder, folder, 'working', 'filter.tif'), 'w', **old_meta) as dst:
        dst.write(new_data, 1)
