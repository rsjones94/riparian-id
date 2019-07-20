import os

import rasterio
from rasterio.merge import merge

parent_folder = r'D:\SkyJones\lidar\2012_tn\system2_overlap\raster_mosaics'
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

