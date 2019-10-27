import sys
sys.path.append(r'C:\Program Files (x86)\GDAL\gdal\python')
import os
os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'

import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gdal
import glob
import sqlite3
from gdal import ogr


par = r'F:\gen_model'
sas = pd.read_excel(os.path.join(par, r'study_areas.xlsx'), dtype={'HUC12': object})
sas = sas.set_index('HUC12')

skippers = ['digsm', 'digel']
if_exists = 'replace' # fail or replace (what to do if the table exists already in the DB). if fail, skips that table/huc

parent = r'F:\gen_model\study_areas'
subs = ['080102040304'] # which HUCS to extract data from

training_folder = r'F:\gen_model\training_sets'

####

code_file = os.path.join(training_folder, 'class_codes.xlsx')
codes = pd.read_excel(code_file)
code_dict = {code:num for code,num in zip(codes['t_code'],codes['n_code'])}

start = time.time()
n_subs = len(subs)

db_loc = os.path.join(training_folder, 'training.db')
conn = sqlite3.connect(db_loc)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
present_tables = [f[0] for f in cursor.fetchall()]
cursor.close()

for k,sub in enumerate(subs):
    if sub in present_tables:
        print(f'{sub} already present in database, if_exists == "{if_exists}"')
        if if_exists != 'replace':
            print(f'ignoring')
            continue
        else:
            print(f'overwriting')
    print(f'Generating training data for {sub}')

    intermediate1 = time.time()

    working = os.path.join(parent,sub)
    train_folder = os.path.join(training_folder,sub)
    train_txt = os.path.join(train_folder, 'training_list.txt')

    # need to make a list of files to use to create the vrt
    band_dict = {}
    with open(train_txt, "w+") as f:
        mosaic_folder = os.path.join(working, r'study_LiDAR\products\mosaic')
        wild = glob.glob(os.path.join(mosaic_folder, r'*.tif'))
        for i,file in enumerate(wild):
            f.write(file+'\n')
            file_only = os.path.basename(os.path.normpath(file))
            band_dict[i+1] = file_only
        i += 1

        class_file = os.path.join(training_folder,sub,r'classification.tif')
        f.write(class_file)
        file_only = os.path.basename(os.path.normpath(class_file))
        band_dict[i + 1] = file_only

    # take the vector shapefile and convert the alphabetic codes to numeric codes
    print('Mapping codes')
    vector_classes = os.path.join(train_folder, r'classification.shp')
    new_field_name = 'Classvalue'
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(vector_classes, 1)
    layer = dataSource.GetLayer()
    new_field = ogr.FieldDefn(new_field_name, ogr.OFTInteger)

    field_names = [field.name for field in layer.schema]

    if new_field_name in field_names:
        layer.DeleteField(field_names.index(new_field_name))
    layer.CreateField(new_field)

    for feature in layer:
        alpha_code = feature.GetField('Classname')
        num_code = code_dict[alpha_code]
        feature.SetField(new_field_name, num_code)
        layer.SetFeature(feature)

    dataSource = None

    # convert the vector classes to raster
    rasterize_command = f'gdal_rasterize -tr 1 1 -a Classvalue -a_nodata 0 {vector_classes} {class_file}'
    print(f'Rasterizing vector classes: {rasterize_command}')
    os.system(rasterize_command)

    # get dims of the training class file, which will be our minimum bounding box

    src = gdal.Open(class_file)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    src = None

    training_vrt = os.path.join(train_folder, "training.vrt")
    vrt_command = f'gdalbuildvrt -separate -te {ulx} {lry} {lrx} {uly} -input_file_list {train_txt} {training_vrt}'
    print(f'Generating VRT: {vrt_command}')
    os.system(vrt_command)

    img = gdal.Open(training_vrt)
    ulx, xres, xskew, uly, yskew, yres = img.GetGeoTransform()
    transformation_file = os.path.join(train_folder, "transformation_key.csv")
    # we need to make a file that tells us how to interconvert between a flattened csv and a geotif
    trans_dict = {
                  'ulx': ulx,
                  'xres': xres,
                  'xskew': xskew,
                  'uly': uly,
                  'yskew': yskew,
                  'yres': yres,
                  'nx': img.RasterXSize,
                  'ny': img.RasterYSize,
                  'epsg': sas.loc[sub].EPSG
                  }
    trans_ser = pd.Series(trans_dict)
    trans_ser.to_csv(transformation_file, header=False)

    n_bands = len(band_dict.keys())
    mils_of_pts = trans_dict["nx"] * trans_dict["ny"] / 10 ** 6
    print(f'Output contains {n_bands} columns with {round(mils_of_pts,3)} million points each'
          f'\nTotal points: {round(n_bands*mils_of_pts,3)} million')

    band_vals = {}
    band_nodatas = {}
    for band in range(img.RasterCount):
        band += 1
        band_name = band_dict[band][:-4]
        if band_name in skippers:
            print(f'Skipping {band_name}')
            continue

        print(f'Flattening {band_name}')
        input_band = img.GetRasterBand(band)
        band_nodatas[band_name] = input_band.GetNoDataValue()
        input_array = np.array(input_band.ReadAsArray())
        flat_array = input_array.flatten()
        band_vals[band_name] = flat_array

    print('Generating dataframe')
    out_data = pd.DataFrame(band_vals)
    out_data['huc12'] = sub
    print('Paring dataframe')
    orig_rows = len(out_data)
    query = ''
    for key,val in band_nodatas.items():
        query += f'{key} != {val} and '
    query = query[:-5]
    print(f'Query: {query}')
    out_data.query(query, inplace=True)
    pared_rows = len(out_data)
    print(f'Dataframe pruned from {orig_rows} rows to {pared_rows} rows (reduced to '
          f'{round(pared_rows/orig_rows*100,2)}% of original)')

    print('Writing to DB')
    out_data.to_sql(name=sub, con=conn, index=True, index_label='cellno', chunksize=50000, if_exists=if_exists)
    print('Establishing index')
    c = conn.cursor()
    c.execute(f"CREATE UNIQUE INDEX idx_cellno ON '{sub}' (cellno)")
    c.close()

    intermediate2 = time.time()
    intermediate_elap = round(intermediate2 - intermediate1, 2)  # in seconds
    running_time = round(intermediate2 - start, 2) / 60  # in minutes
    frac_progress = (k + 1) / n_subs
    estimated_total_time = round(running_time * (1 / frac_progress) - running_time, 2)

    print(f'Processing time: {round(intermediate_elap/60, 2)} minutes. Folder {k + 1} of {n_subs} complete. Estimated time remaining: '
          f'{estimated_total_time} minutes')

conn.close()
final = time.time()
elap = round(final-start, 2)
print(f'FINISHED. Elapsed time: {elap/60} minutes')