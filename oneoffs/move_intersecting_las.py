import os

import whitebox
wbt = whitebox.WhiteboxTools()


parent = r'D:\SkyJones\gen_model\study_areas'
subs = os.listdir(parent)

for sub in subs:
    print(f'\n\n!!!!!!!!!!!!!!!\n Working on {sub} \n!!!!!!!!!!!!!!!\n\n')
    working = os.path.join(parent,sub)
    study = os.path.join(working,'study_area','study_area_r.shp')
    lidar_folder = os.path.join(working,'LiDAR')

    possible = os.listdir(lidar_folder)
    year_folder = [i for i in possible if '20' in i]
    assert len(year_folder) == 1
    all_las = os.path.join(lidar_folder,year_folder[0],'las')
    target = os.path.join(working,'study_LiDAR','las')

    wbt.select_tiles_by_polygon(all_las, target, study)

print('Finished')
