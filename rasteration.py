import os
import time
import sys
from copy import copy
import shutil
import glob
import gdal

import whitebox
import laspy

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


def rasteration(data_folder, products_folder, resolution=1, remove_buildings=True):
    """
    Takes every las file in the data_folder and creates 9 raster products from it (and
    a swapped xyrz las file).
    Each las gets its own subfolder

    Args:
        data_folder: the directory that contains the las files
        products_folder: the directory to write the products to
        resolution: the resolution of the products to be written
        remove_buildings: a boolean indicating if buildings should not be included in the dsm (and dhm) assuming
                          that they are classified in the las file
    Returns: None

    """

    files = os.listdir(data_folder)

    if not isinstance(files, list):
        files = [files]
    n_files = len(files)

    start = time.time()
    problems = []
    for i,file in enumerate(files):
        intermediate1 = time.time()
        filename = os.path.join(data_folder, file)
        new_folder = os.path.join(products_folder, file)[:-4]  # sans .las
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        else:
            print(f'{file} exists. Skipping....')
            continue
        outname = os.path.join(new_folder, file)[:-4]  # sans .las

        print(f'Creating products for {file}')
        block_print()  # wbt has EXTREMELY obnoxious printouts


        # use laspy to store the return number in the user_data field
        try:
            with laspy.file.File(filename, mode="rw") as in_file:
                in_file.user_data = copy(in_file.return_num)
        except TypeError:
            print(f'Problem opening file {file}. Skipping....')
            raise Exception(f'NOOOO {file}')
            os.rmdir(new_folder)
            problems.append(file)
            continue


        # make the digital elevation model (trees, etc. removed)
        # only keep ground road water
        demname = outname+'_digel.tif'
        wbt.lidar_tin_gridding(i=filename, output=demname, parameter='elevation', returns='last', resolution=resolution,
                               exclude_cls=keep([2,9,11]))

        # make the digital surface model
        # keep everything except noise and wires and maybe buildings
        dsmname = outname+'_digsm.tif'
        if remove_buildings:
            cls = '6,7,13,14,18'
        else:
            cls = '7,13,14,18'
        wbt.lidar_tin_gridding(i=filename, output=dsmname, parameter='elevation', returns='first', resolution=resolution,
                               exclude_cls=cls)

        # make the digital height model
        dhmname = outname+'_dighe.tif'
        wbt.subtract(dsmname, demname, dhmname)

        """
        # avg intensity (all returns)
        # keep everything except noise and wires
        allintensname = outname+'_allintens.tif'
        wbt.lidar_tin_gridding(i=filename, output=allintensname, parameter='intensity', returns='all',
                               resolution=resolution,
                               exclude_cls='7,13,14,18')

        # avg intensity (first returns)
        # keep everything except noise and wires
        firstintensname = outname+'_firstintens.tif'
        wbt.lidar_tin_gridding(i=filename, output=firstintensname, parameter='intensity', returns='first', resolution=resolution,
                               exclude_cls='7,13,14,18')
        """

        # make the DEM slope raster
        demslopename = outname+'_demsl.tif'
        wbt.slope(dem=demname, output=demslopename)

        # make the DSM slope raster
        dsmslopename = outname+'_dsmsl.tif'
        wbt.slope(dem=dsmname, output=dsmslopename)

        """
        # make the 1st return intensity slope raster
        firstintensslopename = outname + '_firstintensslope.tif'
        wbt.slope(firstintensname,firstintensslopename)
        """


        # make a tif of the avg number of returns in a cell
        """
        wbt.lidar_point_stats(i=filename, resolution=resolution, num_points=True, num_pulses=True)
        pulse_file = filename[:-4] +'_num_pulses.tif'
        pt_file = filename[:-4] + '_num_pnts.tif'
        nreturnsname = outname + '_nretu.tif'
        wbt.divide(pulse_file, pt_file, nreturnsname)
        os.remove(pulse_file)
        os.remove(pt_file)
        """

        nreturnsname = outname + '_nretu.tif'
        wbt.lidar_tin_gridding(i=filename, output=nreturnsname, parameter='user data', returns='last', resolution=resolution,
                               exclude_cls='7,13,14,18')

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
    print(f'Problem files:')
    print(problems)


def copy_tiles(data_folder, target_folder):
    """
    Takes a folder full of subfolders full of rasters, then creates a subfolder in the target folder and
    copies all appropriate rasters to that subfolder.

    Args:
        data_folder: the folder full of data subfolders
        target_folder: the parent folder that the new subdirectories will be put in

    Returns:
        None
    """

    subs = os.listdir(data_folder)

    if not isinstance(subs, list):
        subs = [subs]
    n_subs = len(subs)

    start = time.time()
    for i,sub in enumerate(subs):
        intermediate1 = time.time()
        print(f'Copying products from {sub}')
        current = os.path.join(data_folder,sub)
        files = os.listdir(current)
        file_types = [f[-9:-4] for f in files]
        for file,ftype in zip(files,file_types):
            target = os.path.join(target_folder,ftype)
            if not os.path.exists(target):
                print(f'MAKING {target}')
                os.mkdir(target)
            to_copy = os.path.join(current,file)
            to_write = os.path.join(target,file)
            if not os.path.exists(to_write):
                shutil.copyfile(to_copy, to_write)
            else:
                print(f'{file} already present in target folder. Skipping....')
                continue

        intermediate2 = time.time()
        intermediate_elap = round(intermediate2-intermediate1, 2) # in seconds
        running_time = round(intermediate2-start, 2)/60 # in minutes
        frac_progress = (i+1)/n_subs
        estimated_total_time = round(running_time*(1/frac_progress) - running_time, 2)

        print(f'Processing time: {intermediate_elap} seconds. Folder {i+1} of {n_subs} complete. Estimated time remaining: '
              f'{estimated_total_time} minutes')

    final = time.time()
    elap = round(final-start, 2)
    print(f'FINISHED. Elapsed time: {elap/60} minutes')


def mosaic_folders(parent, cut_fol, shpf, spatial_ref, path_to_gdal=r'C:\OSGeo4W64\bin'):
    """
    Mosaics all images in every subfolder in the parent folder. Writes output to the parent folder. Also cuts the mosaic
    with a shapefile and copies it to the target folder

    Args:
        parent: the parent directory
        cut_fol: where to copy cuts to
        shpf: shapefile to cut with
        spatial_ref: EPSG code
        path_to_gdal: the path to you GDAL bin

    Returns:
        None

    """
    # path_to_gdal=r'C:\OSGeo4W64\bin'
    wd = os.getcwd()
    subs = [f.name for f in os.scandir(parent) if f.is_dir()]

    if not isinstance(subs, list):
        subs = [subs]
    n_subs = len(subs)

    start = time.time()
    for i,sub in enumerate(subs):
        intermediate1 = time.time()
        print(f'Mosaicing {sub}')

        current = os.path.join(parent,sub)
        os.chdir(current)
        files = os.listdir()
        pairs = []
        out_loc = os.path.join(parent,f'{sub}.tif')

        if not os.path.exists(out_loc):
            try:
                for j, f in enumerate(files):
                    os.rename(f, f'{j}.tif')
                    pairs.append((f, f'{j}.tif'))

                wild = glob.glob('*.tif')
                files_string = " ".join(wild)
                gdal_merge = os.path.join(path_to_gdal, 'gdal_merge.py')

                mosaic_command = f"{gdal_merge} -o {out_loc} -a_nodata -9999 -of gtiff " + files_string
                print(f'Run mosaic command: {mosaic_command}')
                os.system(mosaic_command)

                gdal_edit = os.path.join(path_to_gdal, 'gdal_edit.py')
                edit_command = f'{gdal_edit} -a_srs EPSG:{spatial_ref} {out_loc}'
                print(f'Run edit command: {edit_command}')
                os.system(edit_command)

                stat_command = f'gdalinfo -stats {out_loc}'
                print(f'Run stats command: {stat_command}')
                os.system(stat_command)

                for previous, current in pairs:
                    os.rename(current, previous)

            except KeyboardInterrupt:
                print(f'Keyboard interruption. Repairing file names before termination.')
                for previous, current in pairs:
                    os.rename(current, previous)
        else:
            print(f'{out_loc} exists. Skipping mosaic....')


        """
        cut_output = os.path.join(cut_fol,f'{sub}.tif')
        cut_command = f'gdalwarp -srcnodata -9999 -dstnodata -9999 -crop_to_cutline -cutline {shpf} {out_loc} {cut_output}'
        if not os.path.exists(cut_output):
            print(f'Run command: {cut_command}')
            os.system(cut_command)
        else:
            print(f'{out_loc} exists. Skipping cutting....')
        """

        intermediate2 = time.time()
        intermediate_elap = round(intermediate2-intermediate1, 2) # in seconds
        running_time = round(intermediate2-start, 2)/60 # in minutes
        frac_progress = (i+1)/n_subs
        estimated_total_time = round(running_time*(1/frac_progress) - running_time, 2)

        print(f'Processing time: {intermediate_elap} seconds. Folder {i+1} of {n_subs} complete. Estimated time remaining: '
              f'{estimated_total_time} minutes')

    final = time.time()
    elap = round(final-start, 2)
    print(f'FINISHED. Elapsed time: {elap/60} minutes')
    os.chdir(wd)



    #stat_command = f'gdalinfo -stats {out_loc}'
    #print(f'Run command: {stat_command}')

    #gdalwarp - srcnodata < in > -dstnodata < out > -crop_to_cutline - cutline INPUT.shp INPUT.tif OUTPUT.tif