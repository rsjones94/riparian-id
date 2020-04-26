import os
import time
import sys
from collections import Counter

import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sqlite3
from IPython.display import Image
from sklearn import tree
import pydotplus
from joblib import dump, load
import shutil
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib

from misc_tools import invert_dict, prune_duplicate_leaves
from generate_full_predictions import create_predictions_report

matplotlib.use('Agg')

par = r'F:\gen_model'
training_folder = r'F:\gen_model\training_sets'
models_folder = r'F:\gen_model\models'
unsup_folder = r'F:\gen_model\exploratory\unsupervised'
n_rand = None  # number of samples from each table. None for all samples
ignore_cols = ['cellno', 'classification', 'huc12', 'weight']
var_required = 0.8

training_hucs = ['130202090102']  # if None train all
model_name = 'testing'

####

base_folder = os.path.join(unsup_folder, model_name)

if os.path.exists(base_folder):
    shutil.rmtree(base_folder)
os.mkdir(base_folder)

sas = pd.read_excel(os.path.join(par, r'study_areas.xlsx'), dtype={'HUC12': object})
sas = sas.set_index('HUC12')

start_time = time.time()

db_loc = os.path.join(training_folder, 'training.db')
conn = sqlite3.connect(db_loc)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
present_tables = [f[0] for f in cursor.fetchall()]
cursor.close()

if training_hucs is None:
    training_hucs = present_tables

read_tables = {}
for tab in training_hucs:
    print(f'Reading {tab}')
    if n_rand:
        query = f"SELECT * FROM '{tab}' WHERE cellno IN (SELECT cellno FROM '{tab}' ORDER BY RANDOM() LIMIT {n_rand})"
    else:
        query = f"SELECT * FROM '{tab}'"
    df = pd.read_sql(query, conn)
    df['weight'] = 1 / len(df.index)
    read_tables[tab] = df
conn.close()

read_time = time.time()

if training_hucs is None:
    training_hucs = present_tables

read_elap = read_time - start_time
print(f'Data read. Elapsed time: {round(read_elap / 60, 2)} minutes')

print(f'Reforming training data')
training_df_list = [read_tables[huc] for huc in training_hucs]
df = pd.concat(training_df_list, ignore_index=True)

print('Initializing model')
# https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc

keep_cols = [x for x in df.columns if x not in ignore_cols]

print(f'Transforming data')
# https://towardsdatascience.com/dimension-reduction-techniques-with-python-f36ca7009e5c

x = df.loc[:, keep_cols].values
y = df.loc[:,['classification']].values

xt = StandardScaler().fit_transform(x)
xt = pd.DataFrame(xt)

pca = PCA()
x_pca = pca.fit_transform(xt)
x_pca = pd.DataFrame(x_pca)
x_pca.head()

explained_variance = pca.explained_variance_ratio_

xpln = 0
n = 0
while xpln < var_required:
    xpln += explained_variance[n]
    n += 1

print(f'Need first {n} components to explain {round(var_required*100,2)}% of variance')
print(explained_variance)

x_pca = x_pca.iloc[:, 0:n]
x_pca['classification'] = y


"""
print('Plotting')
outname = os.path.join(base_folder, 'pca.png')
fig = plt.figure()
plt.scatter(x_pca[0], x_pca[1], s=0.2, alpha=0.5, c=x_pca['classification'])
plt.legend()
plt.savefig(outname)
"""

"""
print('Clustering')
dbscan = DBSCAN(eps=1, min_samples=len(df)/100/2)
dbscan.fit(df)
"""

print('Finding neighbors')
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(df)
distances, indices = nbrs.kneighbors(df)

print('Plotting')
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
