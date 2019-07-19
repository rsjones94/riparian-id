import os

import rasterio as rio
import gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import whitebox

wbt = whitebox.WhiteboxTools()

data_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\overlap_las'
products_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
files = os.listdir(data_folder)

def keep(nums):
    # starts with numbers 0 through 18
    # Returns a list where each number given in nums is removed from the starting list.
    excludes = [str(i) for i in range(0,19) if i not in nums]
    return ','.join(excludes)


subf = files[100]
if not isinstance(subf, list):
    subf = [subf]

for file in subf:
    filename = os.path.join(data_folder, file)
    new_folder = os.path.join(products_folder, file)
    os.mkdir(new_folder)
    outname = os.path.join(new_folder, file)[:-4] # sans .las

    # make the digital elevation model (trees, etc. removed)
    # only keep ground road water
    demname = outname+'_dem.tif'
    wbt.lidar_tin_gridding(i=filename, output=demname, parameter='elevation',
                           returns='last', resolution=1, exclude_cls=keep([2,9,11]))

    # make the digital surface model
    # keep everything except noise
    dsmname = outname+'_dsm.tif'
    wbt.lidar_tin_gridding(i=filename, output=dsmname, returns='first', resolution=1,
                           exclude_cls='7,18',)

    # make the digital height model
    topname = outname+'_tophat.las'
    wbt.lidar_tophat_transform(i=filename, output=topname, radius=1)
    dhmname = outname + '_dhm.tif'
    wbt.lidar_tin_gridding(i=topname, output=dhmname, parameter='elevation', resolution=1)

    # digital height model


