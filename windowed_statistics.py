import os
os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'

from scipy.ndimage import generic_filter
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import gdal


def despike(values):
    centre = int(values.size / 2)
    avg = np.mean([values[:centre], values[centre+1:]])
    std = np.std([values[:centre], values[centre+1:]])
    if avg + 3 * std < values[centre]:
        return avg
    else:
        return values[centre]


def perc_over_mean(values, no_data_val=None):
    #if no_data_val:
    #    values = values.copy()
    #    values = values[values != no_data_val]
    #    if len(values) == 0:
    #        return no_data_val
    if no_data_val and no_data_val in values:
        return no_data_val
    avg = np.mean(values)
    n_over_mean = sum(np.greater(values,avg))
    perc_over = n_over_mean/len(values)
    return perc_over


def filter_image(path, out, func, window_size=3, ignore_ndv=True, keywords=None, out_data_type=gdal.GDT_Float32):
    """
    Filters and writes an image

    Args:
        path: filepath to image
        out: output path of filtered image
        func: the filtering function
        window_size: window size (must be odd)
        keywords: keyword arguments to pass to the filtering function
        out_data_type: gdal data type

    Returns:

    """

    print('Reading img')
    img = gdal.Open(path)
    ds = img.GetGeoTransform()
    ulx, xres, xskew, uly, yskew, yres = ds
    nx = img.RasterXSize
    ny = img.RasterYSize
    s = nx*ny

    input_band = img.GetRasterBand(1)
    ndv = input_band.GetNoDataValue()
    input_array = np.array(input_band.ReadAsArray())

    print('Filtering')
    if ignore_ndv:
        keywords['no_data_val'] = ndv
    filtered_band = generic_filter(input=input_array, function=func, size=window_size, extra_keywords=keywords)

    print('Writing data')
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(out, nx, ny, 1, out_data_type)
    outdata.SetGeoTransform(img.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(img.GetProjection())  ##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(filtered_band)
    outdata.GetRasterBand(1).SetNoDataValue(ndv)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!
    outdata = None
    band = None
    ds = None

#data = np.random.randint(0,20,(5,5))
#data[0:3, 0:3] = -9999

#gcm = greycomatrix(data,[1],[0,np.pi/4,np.pi/2,np.pi/2*3],5)
#gcm_mean = gcm.mean(3)[:,:,0]
#gcm_trans = gcm_mean+gcm_mean.transpose()
#gcm_p = gcm_trans / gcm_trans.sum()

#correctedData = generic_filter(data, despike, size=3, mode='nearest')

#ped = generic_filter(data, perc_over_mean, size=3, mode='nearest', extra_keywords={'no_data_val':-9999})

"""
in_file = r'F:\gen_model\study_areas\080102040304\study_LiDAR\products\mosaic\nretu.tif'
out = r'F:\gen_model\study_areas\080102040304\study_LiDAR\products\mosaic\nrepo.tif'
ndv = -9999

i = 0
filter_image(path=in_file, out=out, func=perc_over_mean, window_size=3, ignore_ndv=True, keywords={}, out_data_type=gdal.GDT_Float32)
"""
