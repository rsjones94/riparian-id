from math import sqrt

from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt
from scipy import stats

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

# ref : https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
#image = data.hubble_deep_field()[0:500, 0:500]
#image_gray = rgb2gray(image)

im_path = r'F:\entropy_veg\lidar\las_products\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017\USGS_LPC_TN_27County_blk2_2015_2276581SE_LAS_2017_dhm.tif'
image_gray = io.imread(im_path)
image_gray[image_gray > 500] = 0
image_gray[image_gray < 3] = 0

image_gray = image_gray[2500:, 500:2000]
#image_gray = image_gray[500:2000, 4500:6000]
#image_gray = image_gray[3100:3500, 1100:1500]

io.imshow(image_gray)
io.show()



# blobs
print('Computing laplace of gaussian')
#blobs_log = blob_log(image_gray, max_sigma=35, min_sigma=3, num_sigma=10, threshold=2, overlap=.01)
blobs_log = blob_log(image_gray, max_sigma=35, min_sigma=6, num_sigma=10, threshold=2, overlap=.01)
# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
print('Computed')

fig, ax = plt.subplots(1, 1)
# ax.set_title('Laplacian of Gaussian')
ax.imshow(image_gray)
print('Drawing')
for blob in blobs_log:
    y, x, r = blob
    #c = plt.Circle((x, y), 3, color='red', linewidth=1, fill=False)
    c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
    ax.add_patch(c)
ax.set_axis_off()

plt.tight_layout()
plt.show()


y, x = blobs_log[:,0], blobs_log[:,1]
y = 1500-y
# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
print(xmin, xmax, ymin, ymax)
# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)*10e9

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')
for blob in blobs_log:
    ya, xa, r = blob
    #c = plt.Circle((x, y), 3, color='red', linewidth=1, fill=False)
    c = plt.Circle((xa, 1500-ya), 5, color='red', linewidth=1, fill=True)
    ax.add_patch(c)

plt.show()

print('Done')

"""
# watershedding
image = image_gray
distance = ndi.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
"""
