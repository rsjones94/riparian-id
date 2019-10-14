import os
os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'

import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gdal
import glob

path_to_gdal = r'C:\OSGeo4W64\bin'

#####

par = r'E:\gen_model'
sas = pd.read_excel(os.path.join(par, r'study_areas.xlsx'), dtype={'HUC12': object})
sas = sas.set_index('HUC12')

parent = r'E:\gen_model\study_areas'
subs = ['080102040304']

class_nodata_val = 128
data_nodata_val = -9999

for sub in subs:
    working = os.path.join(parent,sub)
    train_folder = os.path.join(working, 'training')
    train_txt = os.path.join(train_folder, 'training_list.txt')

    # need to make a list of files to use to create the vrt
    band_dict = {}
    training_list = os.path.join(train_folder, "training_list.txt")
    with open(training_list, "w+") as f:
        mosaic_folder = os.path.join(working, r'study_LiDAR\products\mosaic')
        wild = glob.glob(os.path.join(mosaic_folder, r'*.tif'))
        for i,file in enumerate(wild):
            f.write(file+'\n')
            file_only = os.path.basename(os.path.normpath(file))
            band_dict[i+1] = file_only
        i += 1
        class_file = os.path.join(working,r'study_area\classification.tif')
        f.write(class_file)
        file_only = os.path.basename(os.path.normpath(class_file))
        band_dict[i + 1] = file_only

    # get dims of the training class file, which will be our minimum bounding box

    src = gdal.Open(class_file)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    src = None

    training_vrt = os.path.join(train_folder, "training.vrt")
    vrt_command = f'gdalbuildvrt -separate -te {ulx} {lry} {lrx} {uly} -input_file_list {training_list} {training_vrt}'
    print(f'Generating VRT: {vrt_command}')
    os.system(vrt_command)

    img = gdal.Open(training_vrt)
    gtf = img.GetGeoTransform()
    input_array = np.array(img.GetRasterBand(1).ReadAsArray())
