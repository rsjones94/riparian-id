import os
from collections import Counter
from random import randint

from laspy.file import File
from laspy.util import LaspyException

os.chdir(r'D:\sky_laz\Dyer_laz')

files = os.listdir()
dates = []
dyer_skip = 0
for i,file in enumerate(files):
    try:
        date = File(file, mode="r-").header.date
        dates.append(date)
    except LaspyException:
        print(f'too large, skipping Dyer {i}')
        dyer_skip += 1
dyer_counts = Counter(dates)


os.chdir(r'D:\sky_laz\Gibson_laz')
files = os.listdir()
dates = []
gibson_skip = 0
for i,file in enumerate(files):
    try:
        date = File(file, mode="r-").header.date
        dates.append(date)
    except LaspyException:
        print(f'too large, skipping Gibson {i}')
        gibson_skip += 1
gibson_counts = Counter(dates)
