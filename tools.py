import os
import shutil
import time

import gdal
from gdalconst import GA_ReadOnly
import rasterio
from rasterio.merge import merge
import whitebox
wbt = whitebox.WhiteboxTools()


def filter_las_by_bounding(dirs, polys, target_dir):
    """
    Copies any LAS that intersects a vector file to a target directory.

    Args:
        dirs: A list of directories containing LAS files
        polys: A vector polygon file
        target_dir: the directory to copy the intersecting LAS files to

    Returns: None

    """

    for directory in dirs:
        wbt.select_tiles_by_polygon(directory, target_dir, polys)


def create_swapped_las(directory, file, target_folder,
                       lastools_bin_location=r'C:\Users\rj3h\Desktop\LAStools\bin'):
    """
    Creates a .las file with the x and n values swapped
    The output name is ALWAYS xynz_swap.las. The output file will only retain
    the x, y, n and z fields (and z and n are swapped)

    Args:
        directory: The directory that contains the file
        file: the filename, including extension
        target_folder: the folder to write the output to
        lastools_bin_location: the path to the lastools bin folder

    Returns: None

    """

    las2txt = os.path.join(lastools_bin_location, 'las2txt')
    full_file = os.path.join(directory, file)
    command = r'-parse xynz -o'
    out_name = os.path.join(target_folder, 'xyzn_orig.txt')
    convert = las2txt+' '+full_file+' '+' '+command+' '+out_name
    os.system(convert)

    txt2las = os.path.join(lastools_bin_location, 'txt2las')
    swap_command = r'-parse xyzn -o'
    swap_out_name = os.path.join(target_folder, 'xynz_swap.las')
    swap = txt2las+' '+out_name+' '+' '+swap_command+' '+swap_out_name
    os.system(swap)
    os.remove(out_name)


def raster_clip_raster(raster_extent, raster_to_clip, raster_out, path_to_gdal=r'C:\OSGeo4W64\bin'):
    """
    Clips a raster to the extent of another raster

    Args:
        raster_extent: the raster whose extent will be used to clip raster_to_clip
        raster_to_clip: the raster that will be clipped
        raster_out: the full filepath, name and extension of the output
        path_to_gdal: the filepath to your gdal bin

    Returns: nothing
    """

    data = gdal.Open(raster_extent, GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    minx = geo_transform[0]
    maxy = geo_transform[3]
    maxx = minx + geo_transform[1] * data.RasterXSize
    miny = maxy + geo_transform[5] * data.RasterYSize
    os.system(os.path.join(path_to_gdal, 'gdal_translate -projwin ') + ' '.join(
        [str(x) for x in [minx, maxy, maxx, miny]]) + f' -of GTiff {raster_to_clip} {raster_out}')


def tile_raster_with_rasters(main_dir, raster_to_tile, in_ext='_nreturns.tif',
                             out_ext='_valtile.tif'):
    """
    Takes a large raster and tiles it using the extent of the rasters in the subfolders
    of main_dir, writing the output to those same subfolders

    Args:
        main_dir: A directory containing subfolders of rasters
        raster_to_tile: The raster you wish to tile to the extent of the subfolder rasters
        in_ext: a filename suffix+extension combo. Every subfolder should have a file
                named subfolder+in_ext in it. This is used to clip the rasters
        out_ext: a filename suffix+extension combo. The output in each subfolder
                 will be called subfolder+out_ext

    Returns: None

    """
    folders = os.listdir(main_dir)
    n = len(folders)
    start = time.time()
    for i, folder in enumerate(folders):
        intermediate1 = time.time()
        working_dir = os.path.join(main_dir, folder)

        raster_clip_raster(os.path.join(working_dir, folder+in_ext),
                           raster_to_tile,
                           os.path.join(working_dir, folder+out_ext))

        intermediate2 = time.time()
        intermediate_elap = round(intermediate2 - intermediate1, 2)  # in seconds
        running_time = round(intermediate2 - start, 2) / 60  # in minutes
        frac_progress = (i + 1) / n
        estimated_total_time = round(running_time * (1 / frac_progress) - running_time, 2)

        print(f'Processing time: {intermediate_elap} seconds. File {i+1} of {n} complete. Estimated time remaining: '
              f'{estimated_total_time} minutes')

    final = time.time()
    elap = final-start
    print(f'FINISHED. Elapsed time: {round(elap/60, 2)} minutes')


def copy_rasters(parent_folder, out_folder, sub_folders=None, exts=None):
    """
    Takes all rasters that can be formed by combining a sub_folder with an extension
    and copies them to a new directory, creating a subfolder for each extension

    Args:
        parent_folder: The parent directory
        sub_folders: A list of subfolders. If None, all subfolders will be used.
        out_folder:
        exts: a list of 'extensions', which are a filename suffix+extension
              e.g., '_dem.tif'.
              The actual extension should always be a tif

    Returns: None

    """
    if sub_folders is None:
        sub_folders = os.listdir(parent_folder)
    # exts will hold all the file paths for each raster type
    if exts is None:
        exts = ['_allintens.tif',
                '_dem.tif',
                '_demslope.tif',
                '_dhm.tif',
                '_dsm.tif',
                '_dsmslope.tif',
                '_firstintens.tif',
                '_firstintensslope.tif',
                '_nreturns.tif'
                ]

    for ext in exts:
        all_files = [os.path.join(parent_folder, folder, folder+ext) for folder in sub_folders]
        new_dir = os.path.join(out_folder, ext[1:-4])  # sans the .tif
        os.mkdir(new_dir)
        for i, file in enumerate(all_files):
            shutil.copyfile(file, os.path.join(new_dir, str(i)+'.tif'))


def mosaic_sub_folders(parent_folder, sub_folders=None):
    """
    Takes all rasters in a subfolder and mosaics them together, writing the output
    to the parent folder. If sub_folders is not specified, every subfolder is mosaiced.
    If sub_folders is not specified, the parent folder must be empty other than the
    subfolders.

    Args:
        parent_folder: The parent directory
        sub_folders: A list of subfolders in the parent directory

    Returns: None

    """
    if sub_folders is None:
        sub_folders = os.listdir(parent_folder)
    for folder in sub_folders:
        print(f'In folder {folder}')
        target_dir = os.path.join(parent_folder, folder)
        all_files = os.listdir(target_dir)
        all_files = [os.path.join(parent_folder, folder, file) for file in all_files]

        opened_files = [rasterio.open(file) for file in all_files]
        print(f'Collected files. Merging....')
        mosaic, out_trans = merge(opened_files)
        print(f'Files merged. Updating info')
        out_meta = rasterio.open(all_files[0]).meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans
                         }
                        )

        print(f'Writing raster')
        output = os.path.join(parent_folder, 'mosaic_' + folder + '.tif')
        with rasterio.open(output, "w", **out_meta) as dest:
            dest.write(mosaic)

