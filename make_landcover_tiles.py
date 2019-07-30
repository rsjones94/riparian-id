import os, time, shutil

from raster_clip_raster import raster_clip_raster

main_dir = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
folders = os.listdir(main_dir)
n = len(folders)

start = time.time()
for i, folder in enumerate(folders):
    intermediate1 = time.time()
    working_dir = os.path.join(main_dir, folder)

    raster_clip_raster(os.path.join(working_dir, folder+'_allintens.tif'),
                       r'D:\SkyJones\naip\usgs\landcover\sieved_500_cleanv1.tif',
                       os.path.join(working_dir, folder+'_valtile.tif'))

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