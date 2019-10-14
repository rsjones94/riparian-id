import os

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sqlite3

from IPython.display import Image
from sklearn import tree
import pydotplus

n_rand = 100000
training_perc = 0.4

####

par = r'E:\gen_model'

db_loc = os.path.join(par, 'training.db')
conn = sqlite3.connect(db_loc)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
present_tables = [f[0] for f in cursor.fetchall()]
cursor.close()

tab = present_tables[0]
print(f'Reading {tab}')
df = pd.read_sql(f"SELECT * FROM '{tab}' WHERE cellno IN (SELECT cellno FROM '{tab}' ORDER BY RANDOM() LIMIT {n_rand})", conn)
conn.close()

print('Data read. Training model')
feature_cols = ['demsl', 'dighe', 'dsmsl', 'nretu']
ex = df[feature_cols]
why = df['classification']

x_train, x_test, y_train, y_test = train_test_split(ex, why, test_size=1-training_perc, random_state=1)
# test size fraction used to test trained model against

# Create Decision Tree classifier object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Train Decision Tree classifier
model = clf.fit(x_train,y_train)
# Predict the response for test dataset
y_pred = model.predict(x_test)

print(f'Accuracy: {round(metrics.accuracy_score(y_test, y_pred)*100,2)}%')

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_cols,
                                class_names='classifcation')
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
# Show graph
Image(graph.create_png())
