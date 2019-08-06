import os
import time
import sys

import whitebox

from preprocessing_tools import create_swapped_las


wbt = whitebox.WhiteboxTools()

# disable printing
def block_print():
    sys.stdout = open(os.devnull, 'w')


# restore printing
def enable_print():
    sys.stdout = sys.__stdout__


def keep(nums):
    # starts with numbers 0 through 18
    # Returns a list where each number given in nums is removed from the starting list.
    excludes = [str(i) for i in range(0,19) if i not in nums]
    return ','.join(excludes)


def rasteration(data_folder, products_folder, resolution=1,
                lbl=r'C:\Users\rj3h\Desktop\LAStools\bin'):
    """
    Takes every las file in the data_folder and creates 9 raster products from it (and
    a swapped xynz las file).
    These products are each put in their own subfolder.

    Args:
        data_folder: the directory that contains the las files
        products_folder: the directory to write the products to
        resolution: the resolution of the products to be written
        lbl: the path to the lastools bin location
    Returns: None

    """

    files = os.listdir(data_folder)

    if not isinstance(files, list):
        files = [files]
    n_files = len(files)

    start = time.time()
    for i,file in enumerate(files):
        intermediate1 = time.time()
        filename = os.path.join(data_folder, file)
        new_folder = os.path.join(products_folder, file)[:-4]  # sans .lass
        os.mkdir(new_folder)
        outname = os.path.join(new_folder, file)[:-4]  # sans .las

        print(f'Working on {file}')
        block_print()  # wbt has EXTREMELY obnoxious printouts

        # make the digital elevation model (trees, etc. removed)
        # only keep ground road water
        demname = outname+'_dem.tif'
        wbt.lidar_tin_gridding(i=filename, output=demname, parameter='elevation', returns='last', resolution=resolution,
                               exclude_cls=keep([2,9,11]))

        # make the digital surface model
        # keep everything except noise and wires
        dsmname = outname+'_dsm.tif'
        wbt.lidar_tin_gridding(i=filename, output=dsmname, parameter='elevation', returns='first', resolution=resolution,
                               exclude_cls='7,13,14,18')

        # make the digital height model
        dhmname = outname+'_dhm.tif'
        wbt.subtract(dsmname, demname, dhmname)

        # avg intensity (all returns)
        # keep everything except noise and wires
        allintensname = outname+'_allintens.tif'
        wbt.lidar_tin_gridding(i=filename, output=allintensname, parameter='intensity', returns='all', resolution=resolution,
                               exclude_cls='7,13,14,18')

        # avg intensity (first returns)
        # keep everything except noise and wires
        firstintensname = outname+'_firstintens.tif'
        wbt.lidar_tin_gridding(i=filename, output=firstintensname, parameter='intensity', returns='first', resolution=resolution,
                               exclude_cls='7,13,14,18')

        # make the DEM slope raster
        demslopename = outname+'_demslope.tif'
        wbt.slope(dem=demname,output=demslopename)

        # make the DSM slope raster
        dsmslopename = outname+'_dsmslope.tif'
        wbt.slope(dem=dsmname,output=dsmslopename)

        # make the 1st return intensity slope raster
        firstintensslopename = outname + '_firstintensslope.tif'
        wbt.slope(firstintensname,firstintensslopename)

        # finally, we'll make a swapped .las file so we can get a raster of the avg number of returns
        create_swapped_las(directory=data_folder, file=file, target_folder=new_folder, lastools_bin_location=lbl)
        nreturnsname = outname+'_nreturns.tif'
        swappedfile = os.path.join(new_folder, 'xynz_swap.las')
        wbt.lidar_tin_gridding(i=swappedfile, output=nreturnsname, parameter='elevation', returns='all', resolution=resolution,
                               exclude_cls=None)

        # timing
        enable_print()
        intermediate2 = time.time()
        intermediate_elap = round(intermediate2-intermediate1, 2) # in seconds
        running_time = round(intermediate2-start, 2)/60 # in minutes
        frac_progress = (i+1)/n_files
        estimated_total_time = round(running_time*(1/frac_progress) - running_time, 2)

        print(f'Processing time: {intermediate_elap} seconds. File {i+1} of {n_files} complete. Estimated time remaining: '
              f'{estimated_total_time} minutes')

    final = time.time()
    elap = round(final-start, 2)
    print(f'FINISHED. Elapsed time: {elap/60} minutes')
