import whitebox


dirs = [r'D:\sky_laz\Dyer_laz', r'D:\sky_laz\Gibson_laz']
polys = r'D:\SkyJones\shp\System2_UTM16N.shp'
target_dir = r'D:\sky_laz\inbounds_laz\overlaps'

wbt = whitebox.WhiteboxTools()

for directory in dirs:
    wbt.select_tiles_by_polygon(directory, target_dir, polys)