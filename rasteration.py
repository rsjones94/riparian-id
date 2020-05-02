import os
os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'
import time
import sys
from copy import copy
import shutil
import glob

import ogr
import gdal
import whitebox
import laspy

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


def rasteration(data_folder, products_folder, resolution=1, remove_buildings=True, overwrite=False):
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
        elif overwrite:
            print(f'{file} exists. Folder will be overwritten.')
            shutil.rmtree(new_folder)
            os.mkdir(new_folder)
        outname = os.path.join(new_folder, file)[:-4]  # sans .las

        print(f'Creating products for {file}')

        demname = outname+'_digel.tif'
        dsmname = outname + '_digsm.tif'
        nreturnsname = outname + '_nretu.tif'

        block_print()  # wbt has EXTREMELY obnoxious printouts

        if not os.path.exists(nreturnsname):
            # use laspy to store the return number in the user_data field
            try:
                with laspy.file.File(filename, mode="rw") as in_file:
                    in_file.user_data = copy(in_file.return_num)
            except TypeError:
                print(f'A TypeError occurred while attempting to modify {file}. Skipping')
                problems.append(file)
                pass
            except ValueError:
                print(f'A ValueError occurred while attempting to modify {file}. Skipping')
                problems.append(file)
                pass
            wbt.lidar_tin_gridding(i=filename, output=nreturnsname, parameter='user data',
                                   returns='last', resolution=resolution, exclude_cls='7,18')  # , exclude_cls='7,13,14,18'

        # make the digital elevation model (trees, etc. removed)
        # only keep ground road water
        if not os.path.exists(demname):
            wbt.lidar_tin_gridding(i=filename, output=demname, parameter='elevation',
                                   returns='last', resolution=resolution, exclude_cls='7,18')  # , exclude_cls=keep([2,9,11])

        # make the digital surface model
        # keep everything except noise and wires and maybe buildings

        if not os.path.exists(dsmname):
            if remove_buildings:
                cls = '6,7,13,14,18'
            else:
                cls = '7,13,14,18'
            wbt.lidar_tin_gridding(i=filename, output=dsmname, parameter='elevation',
                                   returns='first', resolution=resolution, exclude_cls='7,18')  # , exclude_cls=cls

        """
        # make the digital height model
        dhmname = outname+'_dighe.tif'
        if not os.path.exists(dhmname):
            wbt.subtract(dsmname, demname, dhmname)
        """

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

        """
        # make the DEM slope raster
        demslopename = outname+'_demsl.tif'
        if not os.path.exists(demslopename):
            wbt.slope(dem=demname, output=demslopename)

        # make the DSM slope raster
        dsmslopename = outname+'_dsmsl.tif'
        if not os.path.exists(dsmslopename):
            wbt.slope(dem=dsmname, output=dsmslopename)

        # make the DHM slope raster
        dhmslopename = outname+'_dhmsl.tif'
        if not os.path.exists(dhmslopename):
            wbt.slope(dem=dhmname, output=dhmslopename)

        # make a DEM roughness file, kernel width = 11
        demroughname = outname+'_demro.tif'
        if not os.path.exists(demroughname):
            wbt.average_normal_vector_angular_deviation(dem=demname, output=demroughname, filter=11)

        # make a DSM roughness file, kernel width = 11
        dsmroughname = outname+'_dsmro.tif'
        if not os.path.exists(dsmroughname):
            wbt.average_normal_vector_angular_deviation(dem=dsmname, output=dsmroughname, filter=11)

        # make a DHM roughness file, kernel width = 11
        dhmroughname = outname+'_dhmro.tif'
        if not os.path.exists(dhmroughname):
            wbt.average_normal_vector_angular_deviation(dem=dhmname, output=dhmroughname, filter=11)

        """

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


def copy_tiles(data_folder, target_folder, overwrite=False):
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

    if overwrite and os.path.exists(target_folder):
        print(f'Overwriting {target_folder}....')
        shutil.rmtree(target_folder)

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

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
    Mosaics all images in every subfolder in the parent folder. Writes output to the parent folder. Also generates derivatives

    Args:
        parent: the parent directory
        cut_fol: where to copy cuts to
        shpf: shapefile to cut with
        spatial_ref: EPSG code
        path_to_gdal: the path to you GDAL bin

    Returns:
        None

    """
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

                """
                #add spatial ref
                gdal_edit = os.path.join(path_to_gdal, 'gdal_edit.py')
                edit_command = f'{gdal_edit} -a_srs EPSG:{spatial_ref} {out_loc}'
                print(f'Run edit command: {edit_command}')
                os.system(edit_command)

                # calculate stats
                stat_command = f'gdalinfo -stats {out_loc}'
                print(f'Run stats command: {stat_command}')
                os.system(stat_command)
                """

                for previous, current in pairs:
                    os.rename(current, previous)

            except KeyboardInterrupt:
                print(f'Keyboard interruption. Repairing file names before termination.')
                for previous, current in pairs:
                    os.rename(current, previous)
        else:
            print(f'{out_loc} exists. Skipping mosaic....')

        intermediate2 = time.time()
        intermediate_elap = round(intermediate2-intermediate1, 2) # in seconds
        running_time = round(intermediate2-start, 2)/60 # in minutes
        frac_progress = (i+1)/n_subs
        estimated_total_time = round(running_time*(1/frac_progress) - running_time, 2)

        print(f'Processing time: {intermediate_elap} seconds. Folder {i+1} of {n_subs} complete. Estimated time remaining: '
              f'{estimated_total_time} minutes')


    big_derivs(parent)
    calc_stats_and_ref(parent, spatial_ref, path_to_gdal)

    final = time.time()
    elap = round(final-start, 2)
    print(f'FINISHED. Elapsed time: {elap/60} minutes')
    os.chdir(wd)


def calc_stats_and_ref(folder, spatial_ref, path_to_gdal=r'C:\OSGeo4W64\bin'):
    """

    :param folder:
    :param spatial_ref:
    :return:
    """

    files = [f.name for f in os.scandir(folder) if not f.is_dir() and r'.aux' not in f.name]
    files = [os.path.join(folder,f) for f in files]

    for f in files:
        # add spatial ref
        if not os.path.exists(f + '.aux.xml') and 'tif' in f: # if the aux file exists, we have already added the reference and stats
            gdal_edit = os.path.join(path_to_gdal, 'gdal_edit.py')
            edit_command = f'{gdal_edit} -a_srs EPSG:{spatial_ref} {f}'
            print(f'Run edit command: {edit_command}')
            os.system(edit_command)

            # calculate stats
            stat_command = f'gdalinfo -stats {f}'
            print(f'Run stats command: {stat_command}')
            os.system(stat_command)

def generate_haralicks(in_file, out_folder, out_basename, min_val, max_val, n_bin,
                    path_to_gdal=r'C:\OSGeo4W64\bin', path_to_orfeo=r'C:\OTB-7.0.0-Win64\bin'):
    """
    Creates and splits the 8 basic Haralick textures (Energy, Entropy, Correlation, Inverse Difference Moment,
    Inertia, Cluster Shade, Cluster Prominence and Haralick Correlation) from the Orfeo Haralick output and writes each
    to their own image

    Args:
        in_file: the tif to create textures for
        out_folder: folder to write the textures to
        out_basename: the root name that each texture will share. The suffixes are eg, et, co, id, in, cs, cp, hc
        min_val: the minimum value to bin the tif into
        max_val: the maximum value to bin the tif into
        n_bin: the number of bins
        path_to_gdal: the gdal bin that actually works. not the annoying conda gdal that can't load bigtiffs
        path_to_orfeo: path to the orfeo bin

    Returns:
        None

    """
    di = os.getcwd()
    orf_int = os.path.join(out_folder,'orfeo_intermediate.tif')
    os.chdir(path_to_orfeo)
    orfeo_command = f'otbcli_HaralickTextureExtraction -parameters.min {min_val} -parameters.max {max_val} ' \
              f'-parameters.nbbin {n_bin} -in {in_file} -out {orf_int}'
    print(f'Generating Haralick textures: {orfeo_command}')
    os.system(orfeo_command)

    os.chdir(path_to_gdal)
    suffixes = ['eg', 'et', 'co', 'id', 'in', 'cs', 'cp', 'hc']
    for i,suf in zip(range(1,9),suffixes):
        out_file = os.path.join(out_folder,f'{out_basename}{suf}.tif')
        gdal_command = f'gdal_translate -of Gtiff -b {i} {orf_int} {out_file}'
        print(f'Splitting Haralick texture {i}: {gdal_command}')
        os.system(gdal_command)
    os.remove(orf_int)
    os.chdir(di)


def big_derivs(folder):
    """
    Generates derivative data for rasters

    Args:
        folder: path of target folder

    Returns:
        None
    """

    demname = os.path.join(folder, 'digel.tif')
    dsmname = os.path.join(folder, 'digsm.tif')
    nreturnsname = os.path.join(folder, 'nretu.tif')

    dhmname = os.path.join(folder, 'dighe.tif')

    demslopename = os.path.join(folder, 'demsl.tif')
    dsmslopename = os.path.join(folder, 'dsmsl.tif')
    dhmslopename = os.path.join(folder, 'dhmsl.tif')

    demroughnessname = os.path.join(folder, 'demro.tif')
    dsmroughnessname = os.path.join(folder, 'dsmro.tif')
    dhmroughnessname = os.path.join(folder, 'dhmro.tif')
    nreturnsroughnessname = os.path.join(folder, 'nrero.tif')

    dhmenergyname = os.path.join(folder, 'dhmeg.tif')

    dstncname = os.path.join(folder, 'dstnc.tif')

    #nreteturnspercentovermeanname = os.path.join(folder, 'nrepo.tif') # number of returns, percent over mean

    # cubic convolution resampling
    # dhmresamplename = os.path.join(folder, 'dhmcu.tif')
    # nreturnsresamplename = os.path.join(folder, 'nrecu.tif')

    block_print()  # wbt has EXTREMELY obnoxious printouts
    # make the digital height model

    if not os.path.exists(dhmname):
        print(f'Generating DHM')
        wbt.subtract(dsmname, demname, dhmname)
    else:
        print(f'{dhmname} exists. Skipping generation....')

    # make the DEM slope raster

    if not os.path.exists(demslopename):
        print(f'Generating DEM slope')
        wbt.slope(dem=demname, output=demslopename)
    else:
        print(f'{demslopename} exists. Skipping generation....')

    # make the DSM slope raster
    if not os.path.exists(dsmslopename):
        print(f'Generating DSM slope')
        wbt.slope(dem=dsmname, output=dsmslopename)
    else:
        print(f'{dsmslopename} exists. Skipping generation....')

    # make the DHM slope raster
    if not os.path.exists(dhmslopename):
        print(f'Generating DHM slope')
        wbt.slope(dem=dhmname, output=dhmslopename)
    else:
        print(f'{dhmslopename} exists. Skipping generation....')

    enable_print()

    if not os.path.exists(demroughnessname):
        command = f'gdaldem roughness {demname} {demroughnessname}'
        print(f'Run DEM roughness command: {command}')
        os.system(command)
    else:
        print(f'{demroughnessname} exists. Skipping generation....')

    if not os.path.exists(dsmroughnessname):
        command = f'gdaldem roughness {dsmname} {dsmroughnessname}'
        print(f'Run DSM roughness command: {command}')
        os.system(command)
    else:
        print(f'{dsmroughnessname} exists. Skipping generation....')

    if not os.path.exists(dhmroughnessname):
        command = f'gdaldem roughness {dhmname} {dhmroughnessname}'
        print(f'Run DHM roughness command: {command}')
        os.system(command)
    else:
        print(f'{dhmroughnessname} exists. Skipping generation....')

    if not os.path.exists(nreturnsroughnessname):
        command = f'gdaldem roughness {nreturnsname} {nreturnsroughnessname}'
        print(f'Run return roughness command: {command}')
        os.system(command)
    else:
        print(f'{nreturnsroughnessname} exists. Skipping generation....')

    if not os.path.exists(dhmenergyname):
        generate_haralicks(dhmname, folder, 'dhm', 0, 12, 48) # generate textures using dhm, max height of 12m and bins of 0.25m. Was previously 32m and bins of 0.5m
    else:
        print(f'{dhmenergyname} exists. Skipping generation....')


def createBuffer(inputfn, outputBufferfn, bufferDist):
    """
    http://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html?highlight=buffer

    Args:
        inputfn:
        outputBufferfn:
        bufferDist:

    Returns:

    """
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    bufferlyr = outputBufferds.CreateLayer(outputBufferfn, geom_type=ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()

    for feature in inputlyr:
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        outFeature = None


def generate_distance_raster(polylines, support_folder, outname, epsg=None, cut_shape=None, cut_buffer=1000):
    """
    Generates a raster giving the distance from each cell to the nearest feature in a shapefile

    Args:
        polylines: features to calculate distance to
        support_folder: output FOLDER
        epsg: the epsg to reproject the input polyline file to
        cut_shape: an optional raster used to cut down the output file

    Returns:
        None

    """
    if os.path.exists(support_folder):
        print(f'Removing {support_folder}')
        shutil.rmtree(support_folder)
    print(f'Creating {support_folder}')
    os.mkdir(support_folder)

    #buff_name = os.path.join(support_folder, 'buffered_polylines.shp')
    #createBuffer(polylines, buff_name, 1)

    if epsg:
        repro_name = os.path.join(support_folder, 'polylines_repro.shp')
        repro_command = f'ogr2ogr -t_srs EPSG:{epsg} {repro_name} {polylines}'
        print(f'Run command: {repro_command}')
        os.system(repro_command)
        polylines = repro_name

    if cut_shape:
        buffered_cutter_name = os.path.join(support_folder, 'buffered_cutter.shp')
        print('Creating buffered cutting shape')
        createBuffer(cut_shape, buffered_cutter_name, cut_buffer)

        cut_name = os.path.join(support_folder, 'polylines_cut.shp')
        cut_command = f'ogr2ogr -clipsrc {buffered_cutter_name} {cut_name} {polylines}'
        print(f'Run command: {cut_command}')
        os.system(cut_command)
        polylines = cut_name

    rasterized_name = os.path.join(support_folder, 'rasterized_polylines.tif')
    rasterize_command = f'gdal_rasterize -burn 1 -tr 1 1 -ot Byte {polylines} {rasterized_name}'
    print(f'Run command: {rasterize_command}')
    os.system(rasterize_command)

    distance_name = outname
    distance_command = f'gdal_proximity.py {rasterized_name} {distance_name}'
    print(f'Run command: {distance_command}')
    os.system(distance_command)