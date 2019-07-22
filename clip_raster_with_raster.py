import os

import gdal
from gdalconst import GA_ReadOnly

# raster with extent you with to use to clip
raster_extent = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products\16sbe9591\16sbe9591_allintens.tif'

# raster you'll be clipping
raster_to_clip = r'D:\SkyJones\naip\usgs\mosaic\mosaic.tif'
# output raster location and name
raster_out = r'D:\SkyJones\lidar\2012_tn\system2_overlap\las_products\16sbe9591\cliportho.tif'

data = gdal.Open(raster_extent, GA_ReadOnly)
geoTransform = data.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
maxx = minx + geoTransform[1] * data.RasterXSize
miny = maxy + geoTransform[5] * data.RasterYSize
os.system(r'C:\OSGeo4W64\bin\gdal_translate -projwin ' + ' '.join([str(x) for x in [minx, maxy, maxx, miny]]) + f' -of GTiff {raster_to_clip} {raster_out}')