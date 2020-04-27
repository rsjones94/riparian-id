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

#matplotlib.use('Agg')
matplotlib.use('TkAgg')

par = r'F:\gen_model'
training_folder = r'F:\gen_model\training_sets'
models_folder = r'F:\gen_model\models'
unsup_folder = r'F:\gen_model\exploratory\unsupervised'
n_rand = None  # number of samples from each table. None for all samples
ignore_cols = ['cellno', 'classification', 'huc12', 'weight']
code_file = os.path.join(training_folder, 'class_codes.xlsx')
var_required = 0.9
keep_frac = 0.5
epsilon = 0.15
#num = int(1e5)
num = None

training_hucs = ['080102040304']  # if None train all
# training_hucs = None
model_name = 'testing'

####

codes = pd.read_excel(code_file)
code_dict = {code: num for code, num in zip(codes['t_code'], codes['n_code'])}
inv_code_dict = {num: code for code, num in zip(codes['t_code'], codes['n_code'])}
inv_code_dict_full = {num: code for code, num in zip(codes['category'], codes['n_code'])}

inv_plot_dict_color = {num: clr for clr, num in zip(codes['color'], codes['n_code'])}
inv_plot_dict_linestyle = {num: ls for ls, num in zip(codes['linestyle'], codes['n_code'])}

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
del read_tables
del df

df = pd.concat(training_df_list, ignore_index=True, sort=False)
df = df.sample(n=int(len(df)*keep_frac))

print('Initializing model')
# https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc

keep_cols = [x for x in df.columns if x not in ignore_cols]

print(f'Transforming data')
# https://towardsdatascience.com/dimension-reduction-techniques-with-python-f36ca7009e5c

x = df.loc[:, keep_cols].values
y = df.loc[:,['classification']].values
del df
del training_df_list
xt = StandardScaler().fit_transform(x)
del x
xt = pd.DataFrame(xt)

pca = PCA()
x_pca = pca.fit_transform(xt)
del xt
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
print(f'There are {len(x_pca)} entries')

#num = None
if num is not None:
    data = x_pca.sample(n=num)
else:
    data = x_pca

del x_pca
del y

"""
print('Plotting')
outname = os.path.join(base_folder, 'pca.png')
fig = plt.figure()
plt.scatter(x_pca[0], x_pca[1], s=0.2, alpha=0.5, c=x_pca['classification'])
plt.legend()
plt.savefig(outname)
"""

"""

print('Finding neighbors')
#neigh = NearestNeighbors(n_neighbors=2, radius=0.1, leaf_size=500)
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(data.iloc[:, 0:n])
print('Neighbors found')
distances, indices = nbrs.kneighbors(data.iloc[:, 0:n])

print('Sorting')
distances = np.sort(distances, axis=0)
print('Trimming')
distances = distances[:,1]
print('Plotting')
try:
    plt.plot(distances)
    plt.show()
except:
    plt.plot(distances)
    plt.show()
"""

"""
print('Clustering')
dbscan = DBSCAN(eps=1, min_samples=int(len(data)/100/2))
dbscan.fit(data.iloc[:, 0:n])

clusters = dbscan.labels_
data['cluster'] = clusters


colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid',
          'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])


#plt.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.1, c=vectorizer(clusters))
"""
cees = [inv_plot_dict_color[n] if n in [1,2,3,8,14,17] else 'black' for n in data['classification']]
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.3, c=cees)
