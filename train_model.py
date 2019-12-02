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
n_rand = None  # number of samples from each table. None for all samples


model_a = {
    'model_name': 'fullset_depth8_splitPT0025_bin',

    'training_perc': 0.7,  # percent of data to train on
    'min_split': 0.025, # minimum percentage of samples that a leaf must have to exist
    'drop_cols': [],
    # cols not to use as feature classes. note that dem and dsm are already not included (and others depending on how DB was assembled)
    'class_col': 'classification',  # column that contains classification data
    'training_hucs': None, # what HUCS to train on. If None, use all available. Otherwise input is list of strings

    'reclassing': {
        'trees': ['fo', 'li', 'in']
    },  # classes to cram together. If None, take classes as they are

    'ignore': ['wa', 'cr'],  # classes to exclude from the analysis entirely

    'class_weighting': 'balanced',
    # None for proportional, 'balanced' to make inversely proportional to class frequency
    'criterion': 'gini',  # entropy or gini
    'max_depth': 8,  # max levels to decision tree
    'notes':
        """
        This model uses all data to train a binary classification scheme. No Haralick textures
        """
}


model_b = {
    'model_name': 'fullset_depth8_splitPT0025_tern',

    'training_perc': 0.7,  # percent of data to train on
    'min_split': 0.025, # minimum percentage of samples that a leaf must have to exist
    'drop_cols': [],
    # cols not to use as feature classes. note that dem and dsm are already not included (and others depending on how DB was assembled)
    'class_col': 'classification',  # column that contains classification data
    'training_hucs': None, # what HUCS to train on. If None, use all available. Otherwise input is list of strings

    'reclassing': {
        'trees': ['fo', 'li', 'in'],
        'herb_veg': ['rv', 'we']
    },  # classes to cram together. If None, take classes as they are

    'ignore': ['wa', 'cr'],  # classes to exclude from the analysis entirely

    'class_weighting': 'balanced',
    # None for proportional, 'balanced' to make inversely proportional to class frequency
    'criterion': 'gini',  # entropy or gini
    'max_depth': 8,  # max levels to decision tree
    'notes':
        """
        This model uses all data to train a ternary classification scheme. No Haralick textures
        """
}


model_param_list = [model_a, model_b]

####

for mod in model_param_list:
    model_folder = os.path.join(models_folder, mod['model_name'])
    while os.path.exists(model_folder):
        in_command = input(
            f'Model {mod["model_name"]} exists. Specify a new name, [q]uit, [o]verwrite, or [i]nspect model parameters')
        if in_command == 'q':
            raise Exception('User terminated model training')
        elif in_command == 'o':
            print(f'Overwriting {mod["model_name"]}')
            shutil.rmtree(model_folder)
        elif in_command == 'i':
            print('\n')
            for p, x in mod.items():
                print(f'{p}: {x}')
            print('\n')
        else:
            mod['model_name'] = in_command
            model_folder = os.path.join(models_folder, mod['model_name'])
    os.mkdir(model_folder)

    train_txt = os.path.join(model_folder, 'notes.txt')
    with open(train_txt, "w+") as f:
        f.write(mod['notes'])

sas = pd.read_excel(os.path.join(par, r'study_areas.xlsx'), dtype={'HUC12': object})
sas = sas.set_index('HUC12')

start_time = time.time()

db_loc = os.path.join(training_folder, 'training.db')
conn = sqlite3.connect(db_loc)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
present_tables = [f[0] for f in cursor.fetchall()]
cursor.close()

read_tables = {}
for tab in present_tables:
    print(f'Reading {tab}')
    if n_rand:
        query = f"SELECT * FROM '{tab}' WHERE cellno IN (SELECT cellno FROM '{tab}' ORDER BY RANDOM() LIMIT {n_rand})"
    else:
        query = f"SELECT * FROM '{tab}'"
    df = pd.read_sql(query, conn)
    df['weight'] = 1 / len(df.index)
    read_tables[tab]= df
conn.close()

read_time = time.time()

read_elap = read_time - start_time
print(f'Data read. Elapsed time: {round(read_elap / 60, 2)} minutes')

###
for mod in model_param_list:

    train_start = time.time()

    model_name = mod['model_name']

    if mod['training_hucs'] is None:
        training_hucs = present_tables
    else:
        training_hucs = mod['training_hucs']

    training_perc = mod['training_perc']
    drop_cols = mod['drop_cols']
    drop_cols.extend(['cellno', 'classification', 'huc12', 'weight'])
    class_col = mod['class_col']
    reclassing = mod['reclassing']
    ignore = mod['ignore']
    class_weighting = mod['class_weighting']
    criterion = mod['criterion']
    max_depth = mod['max_depth']
    min_split = mod['min_split']

    model_folder = os.path.join(models_folder, model_name)
    print('\n')
    print(f'Initializing model {model_name}')

    print(f'Reforming training data')

    training_df_list = [read_tables[huc] for huc in training_hucs]
    df = pd.concat(training_df_list, ignore_index=True)

    print(f'Remapping')

    co = Counter(df[class_col])
    co = {int(v): val for v, val in co.items()}

    code_file = os.path.join(training_folder, 'class_codes.xlsx')
    codes = pd.read_excel(code_file)
    code_dict = {code: num for code, num in zip(codes['t_code'], codes['n_code'])}
    inv_code_dict = {num: code for code, num in zip(codes['t_code'], codes['n_code'])}

    present_classes = [inv_code_dict[v] for v in list(co.keys())]

    if reclassing is None:
        reclassing = {cat: [code] for cat, code in zip(codes['category'], codes['t_code'])}
        perfect_mapping = True
    else:
        perfect_mapping = False

    # we need to make sure that each key in reclassing maps to at least one code present in the classifier column
    any_in = lambda a, b: any(i in b for i in a)  # tests if any element of a is in b
    reclassing = {cat: code_list for cat, code_list in reclassing.items() if any_in(code_list, present_classes)}

    class_names = list(reclassing.keys())

    class_map = {i + 1: [code_dict[j] for j in l] for i, l in enumerate(reclassing.values())}
    inv_map = invert_dict(class_map)

    ignore_nums = [code_dict[val] for val in ignore]  # classes we will not use to train the model
    df = df[~df[class_col].isin(ignore_nums)]

    print('Training model')
    cols = list(df.columns)
    feature_cols = [i for i in cols if i not in drop_cols]
    why = []
    n_classes = len(class_names)  # number of SPECIFIED classes; there is another, 'other', if reclass is not none
    for y in df[class_col]:
        try:
            why.append(inv_map[y])
        except KeyError:
            why.append(n_classes + 1)

    x_train, x_test, y_train, y_test = train_test_split(df, why, test_size=1-training_perc, random_state=None)
    # test size fraction used to test trained model against

    # Create Decision Tree classifier object
    clf = DecisionTreeClassifier(criterion=criterion,
                                 max_depth=max_depth,
                                 class_weight=class_weighting,
                                 min_samples_leaf=0.05)
    # Train Decision Tree classifier
    model = clf.fit(x_train[feature_cols], y_train, sample_weight=np.array(x_train['weight']))
    prune_duplicate_leaves(model)
    importances = model.feature_importances_
    # Predict the response for test dataset
    y_pred = model.predict(x_test[feature_cols])

    print(f'Accuracy: {round(metrics.accuracy_score(y_test, y_pred) * 100, 2)}%')
    contributions = {feat: f'{round(imp * 100, 2)}%' for feat, imp in zip(feature_cols, importances)}

    if not perfect_mapping:
        class_names.append('other')
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=feature_cols,
                                    class_names=class_names)
    # Draw graph
    graph = pydotplus.graph_from_dot_data(dot_data)
    # Show graph
    # Image(graph.create_png())

    print(f'Finished training model. Writing reports')

    rep_folder = os.path.join(model_folder, 'reports')
    os.mkdir(rep_folder)

    print(f'(general report)')
    create_predictions_report(y_test=y_test, y_pred=y_pred,
                              class_names=class_names,
                              out_loc=os.path.join(rep_folder, f'full_report_{model_name}.xlsx'),
                              wts=np.array(x_test['weight']))

    for shed in training_hucs:
        print(f'({shed} report)')
        mask = x_test['huc12'] == shed
        sub_y_test = [c for c,m in zip(y_test,mask) if m]
        sub_y_pred = [p for p,m in zip(y_pred,mask) if m]
        create_predictions_report(y_test=sub_y_test, y_pred=sub_y_pred,
                                  class_names=class_names,
                                  out_loc=os.path.join(rep_folder, f'{shed}_report_{model_name}.xlsx'),
                                  wts=None)

    ###

    pickle_clf_name = os.path.join(model_folder, 'clf_package.joblib')
    clf_package = (clf, feature_cols)
    dump(clf_package, pickle_clf_name)

    pickle_param_name = os.path.join(model_folder, 'param_package.joblib')
    param_package = mod
    dump(param_package, pickle_param_name)

    decision_tree_pic = os.path.join(model_folder, 'decision_tree.pdf')
    graph.write_pdf(decision_tree_pic)

    extended_reclass_map = reclassing.copy()
    if not perfect_mapping:
        extended_reclass_map['other'] = 'ALL OTHERS'

    name_mapping = {i + 1: na for i, na in enumerate(class_names)}

    meta_txt = os.path.join(model_folder, 'meta.txt')
    with open(meta_txt, "w+") as f:
        written = f"""\
    Decision Tree Classifier, built with sklearn v{sklearn.__version__}, Python v{sys.version_info[0]}.{
        sys.version_info[1]}.{sys.version_info[2]}
        Trained on {', '.join(training_hucs)}
        Weighting type is {class_weighting} and splitting criterion is {criterion}. Max tree depth: {max_depth}
        Training percent: {round(training_perc * 100, 2)}%
        Minimum split: {round(min_split * 100, 2)}%
        Feature columns: {', '.join(feature_cols)}
        Contributions: {contributions}
        Reclassing: {extended_reclass_map}
        Mapping: {name_mapping}
        Ignored classes: {ignore}
        """
        f.write(written)

    train_time = time.time()
    train_elap = train_time - train_start
    print(f'{model_name} trained. Elapsed time: {round(train_elap / 60, 2)} minutes')

final_time = time.time()
all_train_elap = train_time - read_time
total_elap = train_time - start_time

print(
    f'\n'
    f'Complete. Read time: {round(read_elap / 60, 2)}m. Train time: {round(all_train_elap / 60, 2)}m. '
    f'Total time: {round(total_elap / 60, 2)}m')
