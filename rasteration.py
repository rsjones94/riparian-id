import os, time, sys

import whitebox

from lasswap import create_swapped_las

wbt = whitebox.WhiteboxTools()

data_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\overlap_las'
products_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
files = os.listdir(data_folder)

lbl = r'C:\Users\rj3h\Desktop\LAStools\bin' # lastools bin location

RESOLUTION = 1
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


files = files[100:102]
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
    wbt.lidar_tin_gridding(i=filename, output=demname, parameter='elevation', returns='last', resolution=RESOLUTION,
                           exclude_cls=keep([2,9,11]))

    # make the digital surface model
    # keep everything except noise and wires
    dsmname = outname+'_dsm.tif'
    wbt.lidar_tin_gridding(i=filename, output=dsmname, parameter='elevation', returns='first', resolution=RESOLUTION,
                           exclude_cls='7,13,14,18')

    # make the digital height model
    dhmname = outname+'_dhm.tif'
    wbt.subtract(dsmname, demname, dhmname)

    # avg intensity (all returns)
    # keep everything except noise and wires
    allintensname = outname+'_allintens.tif'
    wbt.lidar_tin_gridding(i=filename, output=allintensname, parameter='intensity', returns='all', resolution=RESOLUTION,
                           exclude_cls='7,13,14,18')

    # avg intensity (first returns)
    # keep everything except noise and wires
    firstintensname = outname+'_firstintens.tif'
    wbt.lidar_tin_gridding(i=filename, output=firstintensname, parameter='intensity', returns='first', resolution=RESOLUTION,
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
    wbt.lidar_tin_gridding(i=swappedfile, output=nreturnsname, parameter='elevation', returns='all', resolution=RESOLUTION,
                           exclude_cls=None)

    enable_print()
    # timing
    intermediate2 = time.time()
    elap = round(intermediate2-intermediate1, 2)
    print(f'Processing time: {elap} seconds. File {i+1} of {n_files} complete.')

final = time.time()
elap = round(final-start, 2)
print(f'FINISHED. Elapsed time: {elap/60} minutes')