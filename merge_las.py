import os

import whitebox
wbt = whitebox.WhiteboxTools()

direc = r'D:\SkyJones\lidar\2012_tn\system2_overlap\overlap_las'
files = os.listdir(direc)
files = [os.path.join(direc, file) for file in files]

f1 = files[0:200]
f2 = files[200:400]
f3 = files[400:600]
f4 = files[600:]

f1s = ', '.join(f1)
f2s = ', '.join(f2)
f3s = ', '.join(f3)
f4s = ', '.join(f4)

fs = [f1s, f2s, f3s, f4s]

for i, f in enumerate(fs):
    wbt.lidar_join(f,
                   f'D:\\SkyJones\\lidar\\2012_tn\\system2_overlap\\merged_las\\merged{i}.las')
