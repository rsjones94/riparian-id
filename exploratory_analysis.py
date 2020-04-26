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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from misc_tools import invert_dict, prune_duplicate_leaves
from generate_full_predictions import create_predictions_report


par = r'F:\gen_model'
training_folder = r'F:\gen_model\training_sets'
output_folder = r'F:\gen_model\exploratory'
hucs = None  # huc codes to analyze. if None, analyze all
boring_cols = ['cellno', 'huc12', 'weight', 'classification']  # columns to not do analysis on

code_file = os.path.join(training_folder, 'class_codes.xlsx')


######

codes = pd.read_excel(code_file)
code_dict = {code: num for code, num in zip(codes['t_code'], codes['n_code'])}
inv_code_dict = {num: code for code, num in zip(codes['t_code'], codes['n_code'])}
inv_code_dict_full = {num: code for code, num in zip(codes['category'], codes['n_code'])}

inv_plot_dict_color = {num: clr for clr, num in zip(codes['color'], codes['n_code'])}
inv_plot_dict_linestyle = {num: ls for ls, num in zip(codes['linestyle'], codes['n_code'])}


sas = pd.read_excel(os.path.join(par, r'study_areas.xlsx'), dtype={'HUC12': object})
sas = sas.set_index('HUC12')

start_time = time.time()

db_loc = os.path.join(training_folder, 'training.db')
conn = sqlite3.connect(db_loc)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
present_tables = [f[0] for f in cursor.fetchall()]
cursor.close()

if hucs is None:
    hucs_of_interest = present_tables
else:
    hucs_of_interest = hucs

read_tables = {}
for tab in hucs_of_interest:
    print(f'Reading {tab}')
    query = f"SELECT * FROM '{tab}'"
    df = pd.read_sql(query, conn)
    df['weight'] = 1 / len(df.index)
    read_tables[tab] = df
conn.close()

read_time = time.time()

read_elap = read_time - start_time
print(f'Data read. Elapsed time: {round(read_elap / 60, 2)} minutes')

## master watershed plot
print(f'Reforming data')
training_df_list = [df for shed, df in read_tables.items()]
master_df = pd.concat(training_df_list, ignore_index=True, sort=False)
read_tables['master'] = master_df

"""
# remove anything falling outside 2 standard deviations
for key, df in read_tables.items():
    print(f'Scrubbing outliers: {key}')
    read_tables[key] = df.mask((df - df.mean()).abs() > 2 * df.std())
"""

xlims = {'demro': (-2,23),
           'dhmco': (-2.5,2.5),
           'dhmcp': (-1,1),
           'dhmcs': (-1250,1250),
           'dhmhc': (-1e4,2.5e5),
           'dhmin': (-10,50),
           'dhmro': (-5,5),
           'dighe': (-1,3),
           'dsmro': (-1, 20)
           }

watershed_master = os.path.join(output_folder, 'constant_watershed')
if os.path.exists(watershed_master):
    shutil.rmtree(watershed_master)
os.mkdir(watershed_master)
## plots holding the watershed constant
for shed, df in read_tables.items():
    print(f'Writing density plots for {shed}')
    class_nums = df['classification'].unique()
    shed_folder = os.path.join(watershed_master, shed)
    if os.path.exists(shed_folder):
        shutil.rmtree(shed_folder)
    os.mkdir(shed_folder)
    for col in df:
        if col in boring_cols:
            # print(f'should skip: {col}')
            continue
        else:
            fig = plt.figure()
            plt.title(f'Density plot: {shed}, {col}')
            plt.xlabel(f'{col} value')
            plt.ylabel(f'Density estimation')
            for n in class_nums:
                rows_of_class = df['classification'] == n
                data = df[col].loc[rows_of_class]
                class_name = inv_code_dict[n]
                long_class_name = inv_code_dict_full[n]
                sns.kdeplot(np.array(data),
                            label=long_class_name,
                            alpha=0.5,
                            color=inv_plot_dict_color[n],
                            ls=inv_plot_dict_linestyle[n])
            plt.legend(fontsize='x-small')
            if col in xlims:
                plt.xlim(xlims[col][0], xlims[col][1])
            out_loc = os.path.join(shed_folder,f'signal_{col}.png')
            fig.savefig(out_loc)
            plt.close(fig)

cols = df.columns

classification_master = os.path.join(output_folder, 'constant_classification')
if os.path.exists(classification_master):
    shutil.rmtree(classification_master)
os.mkdir(classification_master)
## plots holding the classification constant
for n, key in inv_code_dict.items():
    print(f'Writing density plots for {key}')
    class_folder = os.path.join(classification_master, key)
    if os.path.exists(class_folder):
        shutil.rmtree(class_folder)
    os.mkdir(class_folder)
    for col in cols:
        if col in boring_cols:
            # print(f'should skip: {col}')
            continue
        else:
            fig = plt.figure()
            plt.title(f'Density plot: {key}, {col}')
            plt.xlabel(f'{col} value')
            plt.ylabel(f'Density estimation')
            for shed, df in read_tables.items():
                rows_of_class = df['classification'] == n
                data = df[col].loc[rows_of_class]
                class_name = inv_code_dict[n]
                sns.kdeplot(np.array(data), label=shed, alpha=0.5)
            plt.legend(fontsize='x-small')
            if col in xlims:
                plt.xlim(xlims[col][0], xlims[col][1])
            out_loc = os.path.join(class_folder, f'signal_{col}.png')
            fig.savefig(out_loc)
            plt.close(fig)

final_time = time.time()
write_elap = final_time - read_time
total_elap = final_time - start_time

print(
    f'\n'
    f'Complete. Read time: {round(read_elap / 60, 2)}m. Write time: {round(write_elap / 60, 2)}m. '
    f'Total time: {round(total_elap / 60, 2)}m')