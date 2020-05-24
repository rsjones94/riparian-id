import os

os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'

import pandas as pd

from rasteration import *

# consider removing excessive scan angles with wbt.filter_lidar_scan_angles()

ovr_tiles = False
ovr_copy = False
ovr_d2strm = True

par = r'F:\gen_model'
sas = pd.read_excel(os.path.join(par, r'study_areas.xlsx'), dtype={'HUC12': object})
sas = sas.set_index('HUC12')

par = os.path.join(par, r'study_areas')

folders = os.listdir(par)
# only keep folders that start with a number
# folders = [f for f in folders if f[0].isdigit()]

failed = []
success = []

total_n = len(folders)
st = time.time()
for i, sub in enumerate(folders):
    print(f'\n\n!!!!!!!!!!!!!!!\n Working on {sub}, {i + 1} of {total_n} \n!!!!!!!!!!!!!!!\n\n')

    if sub != 'missouri_sub':
        continue

    working = os.path.join(par, sub)
    lidar_folder = os.path.join(working, 'LiDAR')

    possible = os.listdir(lidar_folder)
    year_folder = [i for i in possible if '20' in i]
    assert len(year_folder) == 1
    year_folder = year_folder[0]
    data = os.path.join(lidar_folder, year_folder, 'las')
    rasteration_target = os.path.join(working, 'study_LiDAR', 'products', 'tiled')

    rasteration(data, rasteration_target, resolution=1, remove_buildings=False, overwrite=ovr_tiles)
    copy_target = os.path.join(working, 'study_LiDAR', 'products', 'mosaic')
    cut_target = os.path.join(working, 'study_LiDAR', 'products', 'clipped')
    copy_tiles(rasteration_target, copy_target, overwrite=ovr_copy)

    print('Files copied successfully (or they were present already)....')

    cut_shape = os.path.join(working, 'study_area', 'study_area_r.shp')

    ref_code = sas.loc[sub].EPSG
    mosaic_folders(copy_target, None, None, ref_code)

    if ovr_d2strm:
        # make the raster that gives the distance to the nearest stream (as given by the NHD)
        try:
            polyline_file = os.path.join(working, 'study_area', 'flowlines', 'NHDFlowline.shp')
            cut_shape_file = os.path.join(working, 'study_area', 'study_area_r.shp')
            target_folder = os.path.join(working, 'study_area', 'flowlines', 'distance_to_stream')

            target_name = os.path.join(copy_target, 'dstnc.tif')
            if os.path.exists(target_name):
                os.remove(target_name)
            generate_distance_raster(polyline_file, target_folder,
                                     target_name, epsg=sas.loc[sub].EPSG,
                                     cut_shape=cut_shape_file, cut_buffer=2500)

            success.append(sub)
        except:
            print(f'Failed trying to make dstnc for {sub}')
            failed.append(sub)

en = time.time()

print(f'Data generation complete. Elapsed time: {round((en - st) / 60 / 60, 2)} hours.')
if ovr_d2strm and failed:
    print(f'Failed generating dstnc for {failed}')
    print(f'Successfully generated dstnc for {success}')
