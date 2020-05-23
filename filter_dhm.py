import os

os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'
import itertools
import time

from scipy import ndimage
import ogr
import gdal
import numpy as np

from skimage import measure


def isolated_feature_filter(a):
    """
    Removes isolated pixels. Meant to be used with a 3x3 window

    Args:
        a: the flattened form of an n x m numpy array

    Returns:
        either 0 or the pixel value

    """

    edge_length = int(len(a) ** 0.5)

    b = a.reshape((edge_length, edge_length))

    center = int((edge_length - 1) / 2)
    center_val = b[center, center]

    if center_val <= 0:  # if the value is 0 we can just move on. If it's less than 0 (should not happen in a dhm) then repair it
        #print('Pixel is gucci')
        return 0

    #print('Casting')
    #print(b)
    b = b > 0  # cast to Bools. If DHM is over 0, True
    #print(b)

    if not b.sum() > 1:  # if there are no neighboring pixels with DHM over 0
        #print('Removing')
        return 0
    else:
        #print('Pixel passed muster')
        return center_val


def density_filter(a, thresh=0.3):
    """
    Only keep pixels if over thresh% of pixels in the window are > 0

    Args:
        a: the flattened form of an n x m numpy array
        thresh: filtering threshold

    Returns:
        either 0 or the pixel value


    """
    edge_length = int(len(a) ** 0.5)

    b = a.reshape((edge_length, edge_length))

    center = int((edge_length - 1) / 2)
    center_val = b[center, center]

    if center_val <= 0:  # if the value is 0 we can just move on. If it's less than 0 (should not happen in a dhm) then repair it
        return 0

    b = b > 0  # cast to Bools. If DHM is over 0, True

    density = b.sum() / edge_length**2

    if density >= thresh:
        return center_val
    else:
        return 0


def linear_feature_filter(a):
    """
    Removes linear features based on the algorithm described in

    Characterizing urban surface cover and
    structure with airborne lidar technology
    Nicholas R. Goodwin, Nicholas C. Coops, Thoreau Rory Tooke, Andreas Christen,
    and James A. Voogt

    Args:
        a: the flattened form of an n x m numpy array

    Returns:
        either 0 or the pixel value

    """

    edge_length = int(len(a) ** 0.5)

    b = a.reshape((edge_length, edge_length))

    center = int((edge_length - 1) / 2)
    center_val = b[center, center]

    if center_val <= 0:  # if the value is 0 we can just move on. If it's less than 0 (should not happen in a dhm) then repair it
        return 0

    b = b > 0  # cast to Bools. If DHM is over 0, True

    # data = np.array([[x, y, b[x, y]] for x, y in coords if not np.isnan(b[x, y])])

    # measure.profile_line
    # coords = itertools.product(range(edge_length), range(edge_length))  # all coordinates in the matrix
    start_coords = list(itertools.product([0], range(edge_length)))
    start_coords.extend(list(itertools.product(range(1, edge_length - 1), [edge_length - 1])))
    end_coords = [(edge_length - 1 - a, edge_length - 1 - b) for a, b in start_coords]

    n_filled = b.sum()  # total number of nonzero DHM cells

    i = 0
    for start, end in zip(start_coords, end_coords):
        i += 1
        intercepted = measure.profile_line(b, start, end, linewidth=1)
        n_intercepted = intercepted.sum()


        frac_filled_on_line = (n_intercepted / len(intercepted))
        frac_filled_but_not_on_line = (n_filled - n_intercepted) / edge_length ** 2

        # second part of conditional: are there a lot of points aligned linearly?
        # first part of conditional: are there not a lot of surrounding pixels?
        # if both are true, the feature is probably a powerline or building edge
        if frac_filled_but_not_on_line < 40/81 and frac_filled_on_line >= (3.5 / 9):
            print(f'FILTERING PT. N on line: {n_intercepted} out of {len(intercepted)}. {start}, {end}')
            print(f'Checked {i} lines, value squashed')
            return 0

    #print(f'Checked {i} lines, value passed')
    return center_val


def filter_dhm(dhm, out_file, filter_size=9, technique='density'):
    """
    Remove linear and isolated pixels from a DHM

    Args:
        dhm: path to input digital height model
        out_file: output path
        filter_size: edge length of pixel. Must be odd and over 1. 9. Is preferred. Used only for linear feature
            detection; isolated pixel detection window locked at 3x3
        technique: a string indicating the type method of feature removal. 'density' or 'linear'

    Returns:
        Nothing

    """

    start = time.time()

    if filter_size % 2 != 1 and not filt_size > 1:
        raise Exception('Filter size must be odd and greater than 1')

    print(f'Removing erroneous features from DHM: {out_file}')

    print('Reading dhm')
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
    in_array = in_band.ReadAsArray()
    dhm_nodata_val = in_band.GetNoDataValue()
    print(f'NoData: {dhm_nodata_val}')


    filtered_dhm = in_array



    if technique == 'linear':

        print('Removing isolated pixels')
        filtered_dhm = ndimage.filters.generic_filter(filtered_dhm,
                                                      isolated_feature_filter,
                                                      size=(3, 3))
        for i in range(1):
            print(f'Removing linear features: {i+1}')
            filtered_dhm = ndimage.filters.generic_filter(filtered_dhm,
                                                          linear_feature_filter,
                                                          size=(filter_size, filter_size))

        print('Removing isolated pixels... again')
        filtered_dhm = ndimage.filters.generic_filter(filtered_dhm,
                                                      isolated_feature_filter,
                                                      size=(3, 3))

    elif technique == 'density':
        print('Applying density threshold')
        filtered_dhm = ndimage.filters.generic_filter(filtered_dhm,
                                                      density_filter,
                                                      size=(filter_size, filter_size),
                                                      extra_keywords={'thresh': 0.3})

    else:
        raise Exception(f'Technique must be "density" or "linear"')


    print('Writing')
    outdata.GetRasterBand(1).WriteArray(filtered_dhm)
    outdata.GetRasterBand(1).SetNoDataValue(dhm_nodata_val)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!
    outdata = None
    band = None
    ds = None

    print('DHM processing complete')

    final = time.time()
    elap = round(final - start, 2)
    print(f'Processing time: {round(elap / 60, 2)} minutes')


"""
nums = [9]
for i in nums:
    #dhm = r'F:\gen_model\texture_testing\dhm_mod\raw_dhm.tif'
    dhm = r'F:\gen_model\study_areas\100301011309\study_LiDAR\products\mosaic\dighe.tif'
    # is the DSM better? gives nice texture in foresty but single returny areas, though raisins are a problem
    filt_size = i
    out = f'F:\\gen_model\\texture_testing\\dhm_mod\\processed_dhm_filter{filt_size}.tif'

    filter_dhm(dhm, out, filter_size=filt_size, technique='density')
"""
