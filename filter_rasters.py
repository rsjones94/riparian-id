import os

import numpy as np
import rasterio as rio


def num_to_filter(val):
    mapper = {0: filter_all, 1: filter_greater, 2: filter_lesser, 3: filter_between, 4: filter_outside}
    return mapper[round(val)]

def filter_to_num(val):
    mapper = {filter_all: 0, filter_greater: 1, filter_lesser: 2, filter_between: 3, filter_outside: 4}
    return mapper[round(val)]


def filter_all(params_list, target):
    """
    Makes a mask that is 1 for every cell in the target raster

    Args:
        params_list: A list of length two. Does not do anything but must be provided
        target: the raster which the filter will be applied to

    Returns: a 2d numpy array representing the mask
    """
    with rio.open(os.path.join(target), 'r') as src:
        data = src.read(1)
    data.fill(1)
    return data.astype(int)


def filter_greater(params_list, target):
    """
    Makes a mask that is 1 where values in the target raster are greater than the 1st input parameter, and 0 elsewhere.

    Args:
        params_list: A list of length two. The first value is a float that is the lowest value that will make it through
                the filter. The second value is not used and can be anything, but still must be provided.
        target: the raster which the filter will be applied to

    Returns: a 2d numpy array representing the mask
    """
    with rio.open(os.path.join(target), 'r') as src:
        data = src.read(1)
    mask = data >= params_list[0]
    return mask.astype(int)


def filter_lesser(params_list, target):
    """
    Makes a mask that is 1 where values in the target raster are lesser than the 1st input parameter, and 0 elsewhere.

    Args:
        params_list: A list of length two. The first value is a float that is the highest value that will make it through
                the filter. The second value is not used and can be anything, but still must be provided.
        target: the raster which the filter will be applied to

    Returns: a 2d numpy array representing the mask
    """
    with rio.open(os.path.join(target), 'r') as src:
        data = src.read(1)
    mask = data <= params_list[0]
    return mask.astype(int)


def filter_between(params_list, target):
    """
    Makes a mask that is 1 where values in the target raster are between the input parameters, and 0 elsewhere.

    Args:
        params_list: A list of length two. Each entry is a float. The list need not be sorted.
        target: the raster which the filter will be applied to

    Returns: a 2d numpy array representing the mask
    """
    params_list.sort()
    with rio.open(os.path.join(target), 'r') as src:
        data = src.read(1)
    greater_than_lower = data >= params_list[0]
    lesser_than_higher = data <= params_list[1]
    mask = np.logical_and(greater_than_lower, lesser_than_higher)
    return mask.astype(int)


def filter_outside(params_list, target):
    """
    Makes a mask that is 1 where values in the target raster are outside the input parameters, and 0 elsewhere.

    Args:
        params_list: A list of length two. Each entry is a float. The list need not be sorted.
        target: the raster which the filter will be applied to

    Returns: a 2d numpy array representing the mask
    """
    params_list.sort()
    with rio.open(os.path.join(target), 'r') as src:
        data = src.read(1)
    greater_than_higher = data >= params_list[1]
    lesser_than_lower = data <= params_list[0]
    mask = np.logical_or(greater_than_higher, lesser_than_lower)
    return mask.astype(int)


def filter_rasters(params_list, targets, output=None, write=False):
    """
    Creates an output raster that is the result of a chain of logical ANDS of various raster masks created by applying
    filters to target rasters. All of the rasters should have the same extent and resolution and should have one band.

    Args:
        params_list: a list of parameters 3 entries for every target. The first parameter is the filter, a function
                (filter_all, filter_greater, filter_less, filter_between, filter_outside), the second is the first
                filter value and the thirdis the second filter value
                (only is used when the type of filter is 'between' or outside'). If the filter type is a number,
                it will be coerced to an integer and then mapped
                0:filter_all, 1:filter_greater, 2:filter_lesser, 3:filter_between, 4:filter_outside.
                Note that the filters are INCLUSIVE
        targets: a list of rasters for the filters to be applied to. The first three parameters in param are applied to
                 the first entry in targets, the next three are applied to the second entry, and so on
        output: the location, name and extension of the output raster

    Returns: a numpy array where 1s represent a cell that passed all filters and 0s elsewhere
    """

    # first turn the params_list into sublists where each list goes with one target
    plist = params_list.copy()
    if not len(plist) % 3 == 0:
        raise Exception('params_list must be able to broken into sublists of length 3')
    if not len(plist) == len(targets) * 3:
        raise Exception('three parameters required per target')
    plist = [plist[x:x + 3] for x in range(0, len(plist), 3)] # break plist into sublists of len 3
    for sublist in plist:
        print(sublist)
        if isinstance(sublist[0], int) or isinstance(sublist[0], float):
            sublist[0] = num_to_filter(sublist[0])

    # map the target file onto what function and parameters will be applied to it
    evaluation_dict = {target: sublist for target, sublist in zip(targets, plist)}
    # apply the functions
    masks = [plist[0](plist[1:3], target) for target, plist in evaluation_dict.items()]

    # apply the chained logical and
    all_mask = np.logical_and.reduce(masks).astype(int)

    # write the raster, using the first target to grab the spatial info
    with rio.open(os.path.join(targets[0]), 'r') as src:
        metadata = src.profile

    if write:
        metadata['dtype'] = str(all_mask.dtype)
        with rio.open(output, 'w', **metadata) as dst:
            dst.write(all_mask, 1)

    return all_mask

""" 
p1 = [filter_greater, 4, 0]
p2 = [filter_between, 2, 3]
t1 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\16sbe9493\16sbe9493_dhm.tif'
t2 = r'D:\SkyJones\lidar\2012_tn\system2_overlap\16sbe9493\16sbe9493_nreturns.tif'

p1.extend(p2)
targets = [t1, t2]
output = r'D:\SkyJones\lidar\2012_tn\system2_overlap\16sbe9493\FILTERED.tif'

filter_rasters(p1, targets, output)
"""