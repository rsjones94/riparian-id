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

from misc_tools import invert_dict, prune_duplicate_leaves
from generate_full_predictions import create_predictions_report


par = r'F:\gen_model'
training_folder = r'F:\gen_model\training_sets'
models_folder = r'F:\gen_model\models'


start_time = time.time()

db_loc = os.path.join(training_folder, 'training.db')
conn = sqlite3.connect(db_loc)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
present_tables = [f[0] for f in cursor.fetchall()]
cursor.close()

read_tables = {}
# present_tables = ['102901110304', '010500021301', '080102040304']  # FOR TESTING PURPOSES ONLY
for tab in present_tables:
    print(f'Reading {tab}')
    query = f"SELECT classification FROM '{tab}'"
    df = pd.read_sql(query, conn)
    df['weight'] = 1 / len(df.index)  # proportional weighting for each HUC

    present_cols = df.columns
    read_tables[tab] = df
conn.close()

read_time = time.time()

read_elap = read_time - start_time
print(f'Data read. Elapsed time: {round(read_elap / 60, 2)} minutes')

print(f'Reforming training data')
df = pd.concat([read_tables[huc] for huc in present_tables], ignore_index=True)  # this is the master dataset. it contains ALL data

unique_classes = df['classification'].unique()
n_entries = len(df)
entries_per_class = {i: sum(df['classification'] == i) for i in unique_classes}
epc = {i: entries_per_class[i] for i in range(25) if i in entries_per_class}
