import os

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import sqlite3

n_rand = 100000

par = r'E:\gen_model'

db_loc = os.path.join(par, 'training.db')
conn = sqlite3.connect(db_loc)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
present_tables = [f[0] for f in cursor.fetchall()]
cursor.close()

tab = present_tables[0]
df = pd.read_sql(f"SELECT * FROM '{tab}' WHERE cellno IN (SELECT cellno FROM '{tab}' ORDER BY RANDOM() LIMIT {n_rand})", conn)
conn.close()
