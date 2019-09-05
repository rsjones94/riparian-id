import whitebox
wbt = whitebox.WhiteboxTools()

indir = r'D:\SkyJones\gen_veg\sys2_analysis\lidar\2012_tn\system2_overlap\overlap_las'
outdir = r'D:\SkyJones\gen_model\080102040304\LiDAR\2012\las'
polygons = r'D:\SkyJones\gen_model\080102040304\study_area\huc_080102040304.shp'

wbt.select_tiles_by_polygon(indir,outdir,polygons)