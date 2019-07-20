import os, time, sys, shutil

import whitebox

wbt = whitebox.WhiteboxTools()

parent_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
sub_folders = os.listdir(parent_folder)
out_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\raster_mosaics'

# exts will hold all the file paths for each raster type
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
    # all_files = ','.join(all_files)
    # wbt.mosaic(all_files, output=os.path.join(out_folder, 'mosaic'+ext))
