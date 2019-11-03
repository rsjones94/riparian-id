import os
os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'

import pandas as pd
from joblib import dump, load

from generate_full_predictions import predict_cover

hucs = ['080102040304']
model = 'tn_noroof'

###
par = r'F:\gen_model'

sas = pd.read_excel(os.path.join(par, r'study_areas.xlsx'), dtype={'HUC12': object})
sas = sas.set_index('HUC12')

for huc in hucs:
    fol = os.path.join(par, 'study_areas', huc)
    model_folder = os.path.join(par, 'models', model)
    clf, feature_cols = load(os.path.join(model_folder, 'clf_package.joblib'))
    img = predict_cover(fol, os.path.join(model_folder,huc), feature_cols, clf, sas.loc[huc].EPSG)