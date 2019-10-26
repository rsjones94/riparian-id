import os
import time
import sys

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

from generate_full_predictions import predict_cover

n_rand = None # number of samples from each table. None for all sample
training_perc = 0.3
drop_cols = ['cellno','classification','huc12'] # cols not to use as feature classes
#feature_cols = ['demsl', 'dighe', 'dsmsl', 'nretu']
class_col = 'classification'
class_names = ['Other','Field','Natural','Tree'] # 1, 2, 3, 4

model_name = 'tester'
write_model = True

par = r'F:\gen_model'
training_folder = r'F:\gen_model\training_sets'
models_folder = r'F:\gen_model\models'


####
model_folder = os.path.join(models_folder, model_name)

if os.path.exists(model_folder):
    raise Exception(f'Model {model_folder} exists. Specify new name.')

os.mkdir(model_folder)


sas = pd.read_excel(os.path.join(par, r'study_areas.xlsx'), dtype={'HUC12': object})
sas = sas.set_index('HUC12')

start = time.time()

db_loc = os.path.join(training_folder, 'training.db')
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
cols = list(df.columns)
feature_cols = [i for i in cols if i not in drop_cols]
ex = df[feature_cols]
why = df[class_col]

x_train, x_test, y_train, y_test = train_test_split(ex, why, test_size=1-training_perc, random_state=1)
# test size fraction used to test trained model against

# Create Decision Tree classifier object
clf = DecisionTreeClassifier(criterion="entropy",
                             max_depth=4,
                             min_samples_leaf=0.1)
# Train Decision Tree classifier
model = clf.fit(x_train,y_train)
importances = model.feature_importances_
# Predict the response for test dataset
y_pred = model.predict(x_test)

print(f'Accuracy: {round(metrics.accuracy_score(y_test, y_pred)*100,2)}%')
print('\nContributions')
for feat, imp in zip(feature_cols, importances):
    print(f'{feat}: {round(imp*100,2)}%')

dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=feature_cols,
                                class_names=['Other','Field','Natural','Tree'])
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
# Show graph
# Image(graph.create_png())

cf = metrics.confusion_matrix(y_test,y_pred)

df_cm = pd.DataFrame(cf, index=[i for i in [j + 'P' for j in class_names]],
                     columns=[i for i in [j + 'A' for j in class_names]])
print(df_cm)

report = metrics.classification_report(y_test,y_pred, target_names=class_names, output_dict=True)
print(metrics.classification_report(y_test,y_pred, target_names=class_names))
"""
for n in class_names:
    nameP = n+'P'
    nameA = n+'A'

    total_pix = sum(df_cm[nameA])
    perc_of_sample = round(total_pix/len(y_pred)*100,2)
    correct_pix = df_cm[nameA].loc[nameP]
    perc = round(correct_pix/total_pix*100,2)

    print(f'{sum(df_cm[nameA])} ({perc_of_sample}%) {n} pixels, predicted correctly {perc}% of the time')
"""
final = time.time()
elap = final-start
print(f'Finished training model. Elapsed time: {round(elap/60,2)} minutes')

fol = os.path.join(r'F:\gen_model\study_areas', tab)
of = os.path.join(model_folder, tab)

pickle_model_name = os.path.join(model_folder, 'clf.joblib')
dump(clf, pickle_model_name)

if write_model:
    predict_cover(fol, of, feature_cols, pickle_model_name, sas.loc[tab].EPSG)

decision_tree_pic = os.path.join(model_folder, 'decision_tree.pdf')
graph.write_pdf(decision_tree_pic)


meta_txt = os.path.join(model_folder, 'meta.txt')
with open(meta_txt, "w+") as f:
    written = f"""\
Decision Tree Classifier, built with sklearn v{sklearn.__version__}, Python v{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}
    Trained on {', '.join(present_tables)}
    Feature columns: {', '.join(feature_cols)}
    Training percent: {training_perc}''
    n Pixels per table: {n_rand}
    """
    f.write(written)
