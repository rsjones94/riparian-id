import os, time, sys, shutil

import whitebox

wbt = whitebox.WhiteboxTools()

parent_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\raster_mosaics'
sub_folders = os.listdir(parent_folder)


for folder in sub_folders:
    os.chdir(os.path.join(parent_folder, folder))
    all_files = os.listdir(parent_folder)
    all_files = ','.join(all_files)
    wbt.mosaic(all_files, output=os.path.join(parent_folder, 'mosaic_'+folder+'.tif'))
