import os
os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'

import time

import numpy as np
import pandas as pd
import gdal

from rasteration import calc_stats_and_ref

def predict_cover(huc_folder, out_folder, feature_cols, clf, epsg):
    """
    Generates a raster of landcover using a decision tree

    Args:
        huc_folder: the numeral HUC folder with the data you want to use for the prediction
        out_folder: folder that the data will be written to
        feature_cols: the parameters that the model uses
        clf: sklearn decision tree
        epsg: projection of the output data

    Returns:
        nothing

    """

    start = time.time()

    print(f'Generating prediction for {huc_folder}')

    decision_tree = clf

    pred_folder = out_folder
    os.mkdir(pred_folder)

    # need to make a list of files to use to create the vrt
    band_dict = {}
    pred_list = os.path.join(pred_folder, "prediction_list.txt")
    with open(pred_list, "w+") as f:
        mosaic_folder = os.path.join(huc_folder, r'study_LiDAR\products\mosaic')
        files = [os.path.join(mosaic_folder, col+'.tif') for col in feature_cols]
        for i,file in enumerate(files):
            f.write(file+'\n')
            file_only = os.path.basename(os.path.normpath(file))
            band_dict[i+1] = file_only


    pred_vrt = os.path.join(pred_folder, "pred.vrt")
    vrt_command = f'gdalbuildvrt -separate -input_file_list {pred_list} {pred_vrt}'
    print(f'Generating VRT: {vrt_command}')
    os.system(vrt_command)

    img = gdal.Open(pred_vrt)
    ds = img.GetGeoTransform()
    ulx, xres, xskew, uly, yskew, yres = ds
    nx = img.RasterXSize
    ny = img.RasterYSize

    band_vals = {}
    band_nodatas = {}
    submasks = []
    for band in range(img.RasterCount):
        band += 1
        band_name = band_dict[band][:-4]

        print(f'Flattening {band_name}')
        input_band = img.GetRasterBand(band)
        band_nodatas[band_name] = input_band.GetNoDataValue()
        input_array = np.array(input_band.ReadAsArray())
        flat_array = input_array.flatten()

        smask = flat_array != band_nodatas[band_name]
        submasks.append(smask)

        flat_array[flat_array == band_nodatas[band_name]] = np.mean(flat_array) # fill nodata with mean, but could use different val
        band_vals[band_name] = flat_array

    print('Generating mask')
    mask = np.array(list(map(all, zip(*submasks)))) # logical elementwise AND of all sublists

    print('Generating dataframe')
    data = pd.DataFrame(band_vals)

    print('Making predictions')
    y = decision_tree.predict(data)
    masked = y*mask
    resh = np.reshape(masked, (ny,nx))

    print('Writing predictions')
    out = os.path.join(pred_folder, 'prediction.tif')
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(out, nx, ny, 1, gdal.GDT_Int16)
    outdata.SetGeoTransform(img.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(img.GetProjection())  ##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(resh)
    outdata.GetRasterBand(1).SetNoDataValue(0)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!
    outdata = None
    band = None
    ds = None

    print('Calculating statistics and assigning reference')
    calc_stats_and_ref(pred_folder,epsg)

    final = time.time()
    elap = final - start
    print(f'Prediction complete. Elapsed time: {round(elap / 60, 2)} minutes')

