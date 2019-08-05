"""
This script was used to move validation tiles out of a subfolder into the parent
folder and then delete the empty subfolder
"""

import os, shutil

parent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products'
subs = os.listdir(parent)

for i, sub in enumerate(subs):

    n1 = sub+'_valtile.tif'
    n2 = sub + '_valtile.tif.aux.xml'

    t1 = os.path.join(parent, sub, 'cover_tile', n1)
    t2 = os.path.join(parent, sub, 'cover_tile', n2)

    f1 = os.path.join(parent, sub, n1)
    f2 = os.path.join(parent, sub, n2)

    shutil.move(t1, f1)
    shutil.move(t2, f2)

    shutil.rmtree(os.path.join(parent, sub, 'cover_tile'))
    print(i)
