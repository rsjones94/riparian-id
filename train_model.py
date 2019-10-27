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

from misc_tools import invert_dict


model_name = 'tester_imp_and_roofs_bal'

par = r'F:\gen_model'
training_folder = r'F:\gen_model\training_sets'
models_folder = r'F:\gen_model\models'

n_rand = None # number of samples from each table. None for all sample
class_weighting = 'balanced' # None for proportional, 'balanced' to make inversely proportional to class frequency
training_perc = 0.3
drop_cols = ['cellno','classification','huc12'] # cols not to use as feature classes
feature_cols = ['demsl', 'dighe', 'dsmsl', 'nretu']
class_col = 'classification' # column that contains classification data


reclassing = {
              'trees': ['fo', 'li', 'in'],
              'nat_veg': ['rv', 'we'],
              'imperv': ['im'],
              'roof': ['bt', 'be']
              }

#reclassing = None

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

###

print(f'Remapping')

co = Counter(df[class_col])
co = {int(v):val for v,val in co.items()}

code_file = os.path.join(training_folder, 'class_codes.xlsx')
codes = pd.read_excel(code_file)
code_dict = {code:num for code,num in zip(codes['t_code'],codes['n_code'])}
inv_code_dict = {num:code for code,num in zip(codes['t_code'],codes['n_code'])}

present_classes = [inv_code_dict[v] for v in list(co.keys())]

if reclassing is None:
    reclassing = {cat:[code] for cat,code in zip(codes['category'],codes['t_code'])}
    perfect_mapping = True
else:
    perfect_mapping = False

# we need to make sure that each key in reclassing maps to at least one code present in the classifier column
any_in = lambda a, b: any(i in b for i in a) # tests if any element of a is in b
reclassing = {cat:code_list for cat,code_list in reclassing.items() if any_in(code_list,present_classes)}


class_names = list(reclassing.keys())

class_map = {i+1: [code_dict[j] for j in l] for i,l in enumerate(reclassing.values())}
inv_map = invert_dict(class_map)

print('Data read. Training model')
cols = list(df.columns)
feature_cols = [i for i in cols if i not in drop_cols]
ex = df[feature_cols]
why = []
n_classes = len(class_names) # number of SPECIFIED classes; there is another, 'other', if reclass is not none
for y in df[class_col]:
    try:
        why.append(inv_map[y])
    except KeyError:
        why.append(n_classes+1)

x_train, x_test, y_train, y_test = train_test_split(ex, why, test_size=1-training_perc, random_state=1)
# test size fraction used to test trained model against

# Create Decision Tree classifier object
clf = DecisionTreeClassifier(criterion="entropy",
                             max_depth=3,
                             class_weight=class_weighting)
                             #min_samples_leaf=0.1)
# Train Decision Tree classifier
model = clf.fit(x_train,y_train)
importances = model.feature_importances_
# Predict the response for test dataset
y_pred = model.predict(x_test)

print(f'Accuracy: {round(metrics.accuracy_score(y_test, y_pred)*100,2)}%')
print('\nContributions')
for feat, imp in zip(feature_cols, importances):
    print(f'{feat}: {round(imp*100,2)}%')

if not perfect_mapping:
    class_names.append('other')
dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=feature_cols,
                                class_names=class_names)
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

pickle_model_name = os.path.join(model_folder, 'clf_package.joblib')
clf_package = (clf, feature_cols)
dump(clf_package, pickle_model_name)

decision_tree_pic = os.path.join(model_folder, 'decision_tree.pdf')
graph.write_pdf(decision_tree_pic)

extended_reclass_map = reclassing.copy()
if not perfect_mapping:
    extended_reclass_map['other'] = 'ALL OTHERS'

name_mapping = {i+1:na for i,na in enumerate(class_names)}

meta_txt = os.path.join(model_folder, 'meta.txt')
with open(meta_txt, "w+") as f:
    written = f"""\
Decision Tree Classifier, built with sklearn v{sklearn.__version__}, Python v{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}
    Trained on {', '.join(present_tables)}
    Feature columns: {', '.join(feature_cols)}
    Training percent: {round(training_perc*100,2)}%
    n Pixels per table: {n_rand}
    Reclassing: {extended_reclass_map}
    Mapping: {name_mapping}
    """
    f.write(written)
