import os
import time

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sqlite3

from IPython.display import Image
from sklearn import tree
import pydotplus

n_rand = None # number of samples from each table. None for all sample
training_perc = 0.4
feature_cols = ['demsl', 'dighe', 'dsmsl', 'nretu']
class_col = 'classification'

"""

NOTE:
sklearn currently cannot handle categorical DATA in its decision trees. I do not believe, however, that it can't
handle categorical PREDICTIONS. This model has continuous, noninteger data and categorical unordered output.

"""
####
start = time.time()

par = r'E:\gen_model'

db_loc = os.path.join(par, 'training.db')
conn = sqlite3.connect(db_loc)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
present_tables = [f[0] for f in cursor.fetchall()]
cursor.close()

tab = present_tables[0]
print(f'Reading {tab}')
if n_rand:
    query = f"SELECT * FROM '{tab}' WHERE cellno IN (SELECT cellno FROM '{tab}' ORDER BY RANDOM() LIMIT {n_rand})"
else:
    query = f"SELECT * FROM '{tab}'"
df = pd.read_sql(query, conn)
conn.close()

print('Data read. Training model')
ex = df[feature_cols]
why = df[class_col]

x_train, x_test, y_train, y_test = train_test_split(ex, why, test_size=1-training_perc, random_state=1)
# test size fraction used to test trained model against

# Create Decision Tree classifier object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)
# Train Decision Tree classifier
model = clf.fit(x_train,y_train)
importances = model.feature_importances_
# Predict the response for test dataset
y_pred = model.predict(x_test)

print(f'Accuracy: {round(metrics.accuracy_score(y_test, y_pred)*100,2)}%')
print('\nContributions')
for feat, imp in zip(feature_cols, importances):
    print(f'{feat}: {round(imp*100,2)}%')

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_cols,
                                class_names=class_col)
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
# Show graph
# Image(graph.create_png())
out = os.path.join(par,'decision_tree.pdf')
graph.write_pdf(out)

final = time.time()
elap = round(final-start, 2)
print(f'FINISHED. Elapsed time: {elap/60} minutes')
