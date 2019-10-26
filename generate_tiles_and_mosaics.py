import os
os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'

import pandas as pd

from rasteration import *

# consider removing excessive scan angles with wbt.filter_lidar_scan_angles()

# '180500020905'
# ARRA-CA_GoldenGate_2010_000878.las needed to be reexported to las

"""
refs = { # EPSG codes
        '010500021301': 26919,
        '030902040303': 2777,
        '070801050901': 26915,
        '080102040304': 3723,
        '130202090102': 26913,
        '140801040103': 26913,
        '180500020905': 26910
        }
"""

ovr = False

par = r'F:\gen_model'
sas = pd.read_excel(os.path.join(par, r'study_areas.xlsx'), dtype={'HUC12': object})
sas = sas.set_index('HUC12')

par = os.path.join(par,r'study_areas')

folders = os.listdir(par)
total_n = len(folders)
st = time.time()
for i,sub in enumerate(folders):
    print(f'\n\n!!!!!!!!!!!!!!!\n Working on {sub}, {i+1} of {total_n} \n!!!!!!!!!!!!!!!\n\n')
    working = os.path.join(par,sub)
    lidar_folder = os.path.join(working,'LiDAR')

    possible = os.listdir(lidar_folder)
    year_folder = [i for i in possible if '20' in i]
    assert len(year_folder) == 1
    year_folder = year_folder[0]
    data = os.path.join(lidar_folder,year_folder,'las')
    rasteration_target = os.path.join(working,'study_LiDAR','products','tiled')

    rasteration(data, rasteration_target, resolution=1, overwrite=ovr)
    copy_target = os.path.join(working,'study_LiDAR','products','mosaic')
    cut_target = os.path.join(working,'study_LiDAR','products','clipped')
    copy_tiles(rasteration_target, copy_target, overwrite=ovr)

    cut_shape = os.path.join(working,'study_area','study_area_r.shp')

    ref_code = sas.loc[sub].EPSG
    mosaic_folders(copy_target, cut_target, cut_shape, ref_code)

en = time.time()

print(f'Data generation complete. Elapsed time: {round((en-st)/60/60,2)} hours.')
