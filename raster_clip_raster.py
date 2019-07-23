import os

import gdal
from gdalconst import GA_ReadOnly


def raster_clip_raster(raster_extent, raster_to_clip, raster_out, path_to_gdal=r'C:\OSGeo4W64\bin'):
    """
    Clips a raster to the extent of another raster

    Args:
        raster_extent: the raster whose extent will be used to clip raster_to_clip
        raster_to_clip: the raster that will be clipped
        raster_out: the full filepath, name and extension of the output
        path_to_gdal: the filepath to your gdal bin

    Returns: nothing
    """

    data = gdal.Open(raster_extent, GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    minx = geo_transform[0]
    maxy = geo_transform[3]
    maxx = minx + geo_transform[1] * data.RasterXSize
    miny = maxy + geo_transform[5] * data.RasterYSize
    os.system(os.path.join(path_to_gdal, 'gdal_translate -projwin ') + ' '.join(
        [str(x) for x in [minx, maxy, maxx, miny]]) + f' -of GTiff {raster_to_clip} {raster_out}')
