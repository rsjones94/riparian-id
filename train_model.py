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
n_rand = None  # number of samples from each table. None for all samples. only one of n_rand and keep_frac should not be None
keep_frac = None  # fraction of data from each HUC to keep. If None, keep all
exclude_entirely = ['cellno', 'demro', 'dhmco', 'dhmcp', 'dhmcs', 'dhmeg', 'dhmcp',
                    'dhmet', 'dhmhc', 'dhmid', 'dhmin', 'dhmro', 'dsmro', 'nrero', 'nretu'] # used to lower memory requirements of model

model_a = {
    'model_name': 'bin_deep',

    'training_perc': 0.8,  # percent of data to train on
    'min_split': 0.001,  # minimum percentage of samples that a leaf must have to exist
    'drop_cols': [],
    # cols not to use as feature classes. note that dem and dsm are already not included (and others depending on how DB was assembled)
    'class_col': 'classification',  # column that contains classification data
    'training_hucs': ['180500020905',
                      '070801050901',
                      '130202090102',
                      '080102040304',
                      '010500021301',
                      '030902040303',
                      '140801040103'
                      ],  # what HUCS to train on. If None, use all available. Otherwise input is list of strings

    'reclassing': {
        'trees': ['fo', 'li', 'in']
    },  # classes to cram together. If None, take classes as they are

    'ignore': [],  # classes to exclude from the analysis entirely

    'riparian_distance': 30,  # distance from a stream to be considered riparian

    'class_weighting': 'balanced',
    # None for proportional, 'balanced' to make inversely proportional to class frequency
    'criterion': 'gini',  # entropy or gini
    'max_depth': 20,  # max levels to decision tree
    'notes':
        """
        Standard binary classification
        """
}


model_b = {
    'model_name': 'tern_deep',

    'training_perc': 0.8,  # percent of data to train on
    'min_split': 0.001,  # minimum percentage of samples that a leaf must have to exist
    'drop_cols': [],
    # cols not to use as feature classes. note that dem and dsm are already not included (and others depending on how DB was assembled)
    'class_col': 'classification',  # column that contains classification data
    'training_hucs': ['180500020905',
                      '070801050901',
                      '130202090102',
                      '080102040304',
                      '010500021301',
                      '030902040303',
                      '140801040103'
                      ],  # what HUCS to train on. If None, use all available. Otherwise input is list of strings

    'reclassing': {
        'trees': ['fo', 'li', 'in'],
        'rough': ['rv', 'we', 'ue']
    },  # classes to cram together. If None, take classes as they are

    'ignore': [],  # classes to exclude from the analysis entirely

    'riparian_distance': 30,  # distance from a stream to be considered riparian

    'class_weighting': 'balanced',
    # None for proportional, 'balanced' to make inversely proportional to class frequency
    'criterion': 'gini',  # entropy or gini
    'max_depth': 20,  # max levels to decision tree
    'notes':
        """
        Standard ternary classification
        """
}

model_param_list = [model_a, model_b]

####

# first step is housekeeping

for mod in model_param_list:
    model_folder = os.path.join(models_folder, mod['model_name'])
    while os.path.exists(model_folder):  # make sure we don't unintentionally write over old models
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
    with open(train_txt, "w+") as f:  # add notes in that describe the model
        f.write(mod['notes'])

# next step is to read in our data

sas = pd.read_excel(os.path.join(par, r'study_areas.xlsx'),
                    dtype={'HUC12': object})  # I don't remember what sas stands for lol
sas = sas.set_index('HUC12')

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
    if n_rand:
        query = f"SELECT * FROM '{tab}' WHERE cellno IN (SELECT cellno FROM '{tab}' ORDER BY RANDOM() LIMIT {n_rand})"
    else:
        query = f"SELECT * FROM '{tab}'"
    df = pd.read_sql(query, conn)
    df['weight'] = 1 / len(df.index)  # proportional weighting for each HUC
    if keep_frac:
        df = df.sample(frac=keep_frac)

    present_cols = df.columns
    keep_cols = [p for p in present_cols if p not in exclude_entirely]
    df = df[keep_cols]
    read_tables[tab] = df
conn.close()

read_time = time.time()

read_elap = read_time - start_time
print(f'Data read. Elapsed time: {round(read_elap / 60, 2)} minutes')

# now we actually begin running the models
for mod in model_param_list:

    # extract model parameters
    model_name = mod['model_name']

    print('\n')
    print(f'Initializing model {model_name}')
    train_start = time.time()


    if mod['training_hucs'] is None:
        training_hucs = present_tables
    else:
        training_hucs = mod['training_hucs']
    naive_hucs = [table for table in read_tables if table not in training_hucs]

    training_perc = mod['training_perc']
    drop_cols = mod['drop_cols']
    drop_cols.extend(['cellno', 'classification', 'huc12', 'weight', 'dstnc'])
    class_col = mod['class_col']
    reclassing = mod['reclassing']
    ignore = mod['ignore']
    class_weighting = mod['class_weighting']
    criterion = mod['criterion']
    max_depth = mod['max_depth']
    min_split = mod['min_split']
    riparian_distance = mod['riparian_distance']

    model_folder = os.path.join(models_folder, model_name)

    print(f'Reforming training data')  # now we need to create two datasets
    # one dataset is the TRAINED data, i.e., the data from HUCs we will use to train the model (we will also set aside some of that data for internal validation)
    # the other dataset is NAIVE data, i.e., data from HUCs that the model was not trained on at all

    df = pd.concat([read_tables[huc] for huc in present_tables], ignore_index=True)  # this is the master dataset. it contains ALL data
    if len(model_param_list) == 1:
        del read_tables # this is a big chunk of memory that we can free up if we don't need to call it back up again later

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
    df = df[~df[class_col].isin(ignore_nums)]  # drop those rows

    print('Training model')
    cols = list(df.columns)
    feature_cols = [i for i in cols if i not in drop_cols]

    why = []  # why will be a list of class codes remapped to an aggregate code. in other words, why is the response variable
    n_classes = len(class_names)  # number of SPECIFIED classes; there is another, 'other', if reclass is not none
    for y in df[class_col]:
        try:
            why.append(inv_map[y])
        except KeyError:
            why.append(n_classes + 1)

    df['why'] = why

    # training data. use the training hucs
    x_train, x_test, y_train, y_test = train_test_split(df.loc[df['huc12'].isin(training_hucs)],
                                                        df.loc[df['huc12'].isin(training_hucs)]['why'],
                                                        test_size=1 - training_perc,
                                                        random_state=None)

    # naive data. use the naive hucs. note that we actually don't need any training data for the naive set, just testing data
    x_train_naive, x_test_naive, y_train_naive, y_test_naive = train_test_split(df.loc[df['huc12'].isin(naive_hucs)],
                                                                                df.loc[df['huc12'].isin(naive_hucs)]['why'],
                                                                                test_size=0.999,
                                                                                random_state=None)

    # Train Decision Tree classifier
    clf = DecisionTreeClassifier(criterion=criterion,
                                 max_depth=max_depth,
                                 class_weight=class_weighting,
                                 min_samples_leaf=min_split)
    model = clf.fit(x_train[feature_cols], y_train, sample_weight=np.array(x_train['weight'])) # the weighting here ensures that as a whole, each HUC is given the same weight
    # even if the number of samples is different. This is NOT the weighting of each aggregate class
    #model = clf.fit(x_train[feature_cols], y_train)
    prune_duplicate_leaves(model)
    importances = model.feature_importances_
    # Predict the response for test dataset (and for the naive dataset)
    y_pred = model.predict(x_test[feature_cols])
    y_pred_naive = model.predict(x_test_naive[feature_cols])


    print(f'Accuracy (trained): {round(metrics.accuracy_score(y_test, y_pred) * 100, 2)}%')
    print(f'Accuracy (naive): {round(metrics.accuracy_score(y_test_naive, y_pred_naive) * 100, 2)}%')
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

    decision_tree_pic = os.path.join(model_folder, 'decision_tree.pdf')
    graph.write_pdf(decision_tree_pic)

    print(f'Finished training model. Writing reports')
    rep_folder = os.path.join(model_folder, 'reports')
    os.mkdir(rep_folder)

    print(f'(aggregate trained report)')
    create_predictions_report(y_test=y_test, y_pred=y_pred,
                              class_names=class_names,
                              out_loc=os.path.join(rep_folder, f'aggregate_report_{model_name}_TRAINED.xlsx'),
                              wts=x_test['weight'])

    dist_mask_upper = df.loc[df['huc12'].isin(training_hucs)]['dstnc'] > 0
    dist_mask_lower = df.loc[df['huc12'].isin(training_hucs)]['dstnc'] <= riparian_distance
    dist_mask = [a and b for a, b in zip(dist_mask_upper, dist_mask_lower)]
    trained_frac_in_buffer = sum(dist_mask) / len(dist_mask)
    print(f'{round(trained_frac_in_buffer*100, 2)}% within riparian buffer')

    sub_y_test_riparian = [c for c, d in zip(y_test, dist_mask) if d]
    sub_y_pred_riparian = [p for p, d in zip(y_pred, dist_mask) if d]
    weight_riparian = [p for p, d in zip(x_test['weight'], dist_mask) if d]

    create_predictions_report(y_test=sub_y_test_riparian, y_pred=sub_y_pred_riparian,
                              class_names=class_names,
                              out_loc=os.path.join(rep_folder, f'aggregate_report_{model_name}_TRAINED_RIPARIAN.xlsx'),
                              wts=weight_riparian)

    print(f'(aggregate naive report)')
    create_predictions_report(y_test=y_test_naive, y_pred=y_pred_naive,
                              class_names=class_names,
                              out_loc=os.path.join(rep_folder, f'aggregate_report_{model_name}_NAIVE.xlsx'),
                              wts=x_test_naive['weight'])

    dist_mask_upper = df.loc[df['huc12'].isin(naive_hucs)]['dstnc'] > 0
    dist_mask_lower = df.loc[df['huc12'].isin(naive_hucs)]['dstnc'] <= riparian_distance
    dist_mask = [a and b for a, b in zip(dist_mask_upper, dist_mask_lower)]
    naive_frac_in_buffer = sum(dist_mask) / len(dist_mask)
    print(f'{round(naive_frac_in_buffer*100, 2)}% within riparian buffer')

    sub_y_test_riparian = [c for c, d in zip(y_test_naive, dist_mask) if d]
    sub_y_pred_riparian = [p for p, d in zip(y_pred_naive, dist_mask) if d]
    weight_riparian = [p for p, d in zip(x_test_naive['weight'], dist_mask) if d]

    create_predictions_report(y_test=sub_y_test_riparian, y_pred=sub_y_pred_riparian,
                              class_names=class_names,
                              out_loc=os.path.join(rep_folder, f'aggregate_report_{model_name}_NAIVE_RIPARIAN.xlsx'),
                              wts=weight_riparian)

    del df

    for shed in present_tables:

        if shed in naive_hucs:
            add_name = 'NAIVE'
            ex = x_test_naive
            wi_test = y_test_naive
            wi_pred = y_pred_naive
        elif shed in training_hucs:
            add_name = 'TRAINED'
            ex = x_test
            wi_test = y_test
            wi_pred = y_pred
        print(f'({shed} report) - {add_name}')

        huc_mask = ex['huc12'] == shed
        dist_mask_upper = ex['dstnc'] > 0
        dist_mask_lower = ex['dstnc'] <= riparian_distance
        dist_mask = [a and b for a,b in zip(dist_mask_upper, dist_mask_lower)]

        sub_y_test = [c for c,m in zip(wi_test, huc_mask) if m]
        sub_y_pred = [p for p,m in zip(wi_pred, huc_mask) if m]

        try:
            create_predictions_report(y_test=sub_y_test, y_pred=sub_y_pred,
                                      class_names=class_names,
                                      out_loc=os.path.join(rep_folder, f'{shed}_report_{model_name}_{add_name}.xlsx'),
                                      wts=None)
        except AssertionError:
            print(f'Failed to generate report {shed}_report_{model_name}_{add_name}.xlsx')

        sub_y_test_riparian = [c for c, m, d in zip(wi_test, huc_mask, dist_mask) if m and d]
        sub_y_pred_riparian = [p for p, m, d in zip(wi_pred, huc_mask, dist_mask) if m and d]

        try:
            create_predictions_report(y_test=sub_y_test_riparian, y_pred=sub_y_pred_riparian,
                                      class_names=class_names,
                                      out_loc=os.path.join(rep_folder, f'{shed}_report_{model_name}_{add_name}_RIPARIAN.xlsx'),
                                      wts=None)
        except AssertionError:
            print(f'Failed to generate report {shed}_report_{model_name}_{add_name}_RIPARIAN.xlsx'
                  f'\n This may be due to a lack of riparian cells')

    print('PACKAGING MODEL')

    pickle_clf_name = os.path.join(model_folder, 'clf_package.joblib')
    clf_package = (clf, feature_cols)
    dump(clf_package, pickle_clf_name)

    pickle_param_name = os.path.join(model_folder, 'param_package.joblib')
    param_package = mod
    dump(param_package, pickle_param_name)

    pickle_importances_name = os.path.join(model_folder, 'importances_package.joblib')
    importances_package = {feat: imp for feat, imp in zip(feature_cols, importances)}
    dump(importances_package, pickle_importances_name)

    extended_reclass_map = reclassing.copy()
    if not perfect_mapping:
        extended_reclass_map['other'] = 'ALL OTHERS'

    if keep_frac is None:
        keep_frac = 'All'

    name_mapping = {i + 1: na for i, na in enumerate(class_names)}

    meta_txt = os.path.join(model_folder, 'meta.txt')
    with open(meta_txt, "w+") as f:
        written = f"""\
        Decision Tree Classifier, built with sklearn v{sklearn.__version__}, Python v{sys.version_info[0]}.{
        sys.version_info[1]}.{sys.version_info[2]}
            Trained on {', '.join(training_hucs)}
            Naive watersheds: {', '.join(naive_hucs)}
            Weighting type is {class_weighting} and splitting criterion is {criterion}. Max tree depth: {max_depth}
            Training percent: {round(training_perc * 100, 2)}%
            Minimum split: {round(min_split * 100, 2)}%
            Feature columns: {', '.join(feature_cols)}
            Contributions: {contributions}
            Reclassing: {extended_reclass_map}
            Mapping: {name_mapping}
            Ignored classes: {ignore}
            Riparian distance: {riparian_distance} ({round(trained_frac_in_buffer*100,4)}% of trained within buffer. {round(naive_frac_in_buffer*100,4)}% of naive within buffer)
            % of data retained: {keep_frac}
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
