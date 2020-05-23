import os
os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'
import itertools
import time

from scipy import ndimage
import ogr
import gdal
import numpy as np

from fit_plane_to_points import fit_plane
from rasteration import generate_laplace

ex = np.array([[1,2,1],[3,-1,2],[1,1,2]])


def residual_filter(a, nodata_val=None):
    """
    Fits a linear regression to all points in a square matrix excluding the central value of an array. Then returns the residual of
    the central point to the regression plane. Assumes point spacing is constant. The matrix must have odd dimensions

    Args:
        a: the flattened form of an n x m numpy array

    Returns:
        the residual of the central value for a linear regression of all points excluding the central value

    """

    if nodata_val:
        if nodata_val in a:
            return nodata_val

    edge_length = int(len(a)**0.5)

    b = a.reshape((edge_length, edge_length))
    b = b.astype(float)
    coords = itertools.product(range(edge_length), range(edge_length))  # all coordinates in the matrix

    center = int((edge_length-1) / 2)
    center_val = b[center, center]  #  save this - we will get the orthogonal distance to the plane of best fit later
    b[center, center] = np.nan  # remove this point - we will not use it for calculating the plane of best fit

    data = np.array([[x, y, b[x, y]] for x, y in coords if not np.isnan(b[x, y])])

    C = fit_plane(data)

    predicted = C[0]*center + C[1]*center + C[2]
    residual = center_val - predicted # if positive, the center val is above the plane
    return residual


def texturize_dhm(dsm, out_file, filter_size=3, remove_outliers=True, dhm=None):
    """
    Adds the local texture fro ma dsm to a DHM so the DHM can be used more effectively with e.g., Haralick textures.
    That output will share projection and NoData vals with the input dsm. The DHM and dsm must have the same shape

    Args:
        dsm: path to digital surface model from which the DHM was derived
        out_file: output path
        filter_size: size of the textural operator. For example, a size of 3 will filter using a 3x3 window. The filter
        size must be odd.
        remove_outliers: whether or not to filter values outside of a window. If outlier removal is desired,
            pass a list of length two of the form [low_percentile, high_percentile]. For example, [1, 99] will keep
            values only falling between the 1st and 99th percentile.
        dhm: path to input digital height model. If specified, the texture will be layered onto the dhm

    Returns:
        Nothing

    """

    start = time.time()

    if filter_size % 2 != 1 and not filt_size > 1:
        raise Exception('Filter size must be odd and greater than 1')

    print(f'Texturizing: {out_file}')


    print('Reading dsm')
    img = gdal.Open(dsm)
    ds = img.GetGeoTransform()
    ulx, xres, xskew, uly, yskew, yres = ds
    nx = img.RasterXSize
    ny = img.RasterYSize

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(out_file, nx, ny, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(img.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(img.GetProjection())  ##sets same projection as input

    in_band = img.GetRasterBand(1)
    in_array = in_band.ReadAsArray()
    dsm_nodata_val = in_band.GetNoDataValue()
    print(f'NoData: {dsm_nodata_val}')

    print('Filtering')
    filtered_dsm = ndimage.filters.generic_filter(in_array,
                                           residual_filter,
                                           size=(filter_size,filter_size),
                                           mode='nearest',
                                           extra_keywords={'nodata_val': dsm_nodata_val})

    if dhm:
        print('Reading DHM')
        img = gdal.Open(dhm)
        ds = img.GetGeoTransform()
        ulx, xres, xskew, uly, yskew, yres = ds
        nx = img.RasterXSize
        ny = img.RasterYSize

        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(out_file, nx, ny, 1, gdal.GDT_Float32)
        outdata.SetGeoTransform(img.GetGeoTransform())  ##sets same geotransform as input
        outdata.SetProjection(img.GetProjection())  ##sets same projection as input

        in_band = img.GetRasterBand(1)
        out_array = in_band.ReadAsArray()
        dhm_nodata_val = in_band.GetNoDataValue()

        print('Merging DHM and filtered dsm')
        # For every element, use the filtered dsm value if the DHM <= 0, else use the DHM value. If either element is
        # the respective NoData value, write in the dsm NoData value
        out_array[out_array <= 0] = filtered_dsm[out_array <= 0] # generally NoData values are a negative number, so this should convert NoDatas as well
    else:
        out_array = filtered_dsm

    if remove_outliers:
        print('Removing outliers')
        out_array[out_array == dsm_nodata_val] = np.nan

        lowfilt, highfilt = np.nanpercentile(out_array, remove_outliers)
        print(lowfilt, highfilt)

        out_array[out_array < lowfilt] = lowfilt
        out_array[out_array > highfilt] = highfilt

    print('Writing')
    outdata.GetRasterBand(1).WriteArray(out_array)
    outdata.GetRasterBand(1).SetNoDataValue(dsm_nodata_val)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!
    outdata = None
    band = None
    ds = None

    print('Texturization complete')

    final = time.time()
    elap = round(final-start, 2)
    print(f'Texturization time: {round(elap/60,2)} minutes')

"""
nums = [3, 5, 7, 9]
for i in nums:
    dhm = r'F:\gen_model\texture_testing\MT_Helena_2012_000356_dighm.tif'
    #dsm = r'F:\gen_model\study_areas\100301011309\study_LiDAR\products\tiled\MT_Helena_2012_000356\MT_Helena_2012_000356_digsm.tif'
    dsm = r'F:\gen_model\texture_testing\sub\subsubdsm.tif'
    # is the DSM better? gives nice texture in foresty but single returny areas, though raisins are a problem
    filt_size = i
    #out = f'F:\\gen_model\\texture_testing\\detrend_dsm_filt{filt_size}.tif'
    out = f'F:\\gen_model\\texture_testing\\sub\\detrend_dsm_filt{filt_size}_outlier_removal.tif'
    lap_out = f'F:\\gen_model\\texture_testing\\sub\\laplace_dsm.tif'

    #texturize_dhm(dsm=dsm, out_file=out, filter_size=filt_size, remove_outliers=[1,99])
    generate_laplace(dsm, lap_out)
"""
