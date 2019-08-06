import os

import laspy

os.chdir(r'D:\SkyJones\lidar\2012_tn\system2_overlap\test')

files = os.listdir()
for i, name in enumerate(files):
    in_file = laspy.file.File(name, mode="rw")
    out_file = laspy.file.File(str(i)+'.las', header=in_file.header, mode='w')
    out_head = out_file.header
    #out_file.points = in_file.points
    out_file.X, out_file.Y, out_file.Z = in_file.X, in_file.Y, in_file.num_returns
    out_head.update_min_max()
    out_file.close()
