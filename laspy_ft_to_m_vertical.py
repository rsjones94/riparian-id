import os
import time

import laspy

infol = r'F:\gen_model\study_areas\030902040303\LiDAR\2X0X0X7_mh_ft_v\las'
outfol = r'F:\gen_model\study_areas\030902040303\LiDAR\2007_m\las'

os.chdir(infol)

start = time.time()
files = os.listdir()
n_files = len(files)
for i, name in enumerate(files):
    intermediate1 = time.time()
    in_file = laspy.file.File(name, mode="rw")
    out_file = laspy.file.File(os.path.join(outfol, name), header=in_file.header, mode='w')
    out_head = out_file.header
    out_file.points = in_file.points
    out_file.Z = in_file.Z * 0.3048
    out_file.z = in_file.z * 0.3048
    out_head.update_min_max()
    in_file.close()
    out_file.close()

    intermediate2 = time.time()
    intermediate_elap = round(intermediate2 - intermediate1, 2)  # in seconds
    running_time = round(intermediate2 - start, 2) / 60  # in minutes
    frac_progress = (i + 1) / n_files
    estimated_total_time = round(running_time * (1 / frac_progress) - running_time, 2)

    print(f'Processing time: {round(intermediate_elap/60, 2)} minutes. Folder {i + 1} of {n_files} complete. Estimated time remaining: '
          f'{estimated_total_time} minutes')
