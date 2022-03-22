"""
PURPOSE: running shap on matrix data to better understand what features are most
	important

03/09/2022: Ally adjusts code to run on classification rather than regression

USAGE EX: python /mnt/home/peipeiw/Documents/Ath_GS/SHAP/SHAP_training.py -df
/mnt/home/peipeiw/Documents/Ath_GS/Models_for_Grimm_pheno/Input_data
/GSE80744_TPM_383_accessions_loge_plus1.csv -df2
/mnt/home/peipeiw/Documents/Ath_GS/Models_for_Grimm_pheno
/Phenotype_value_383_common_accessions_2017_Grimm.csv -sep "," -y_name
FT10_mean -test /mnt/home/peipeiw/Documents/Ath_GS/Models_for_Grimm_pheno
/Test.txt -save expression -n_estimators 100 -max_depth 3 -max_features 0.5
-n_jobs 8 -top 20 """

import shap
import sys, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datatable as dt
import joblib
from sklearn.ensemble import RandomForestClassifier


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def main():
    parser = argparse.ArgumentParser(
        description='Calculate SHAP values, i.e., the contribution of each '
                    'feature to the prediction of each instance')

    #######################
    ### Input arguments ###
    #######################

    # Required
    req_group = parser.add_argument_group(title='REQUIRED INPUT')
    req_group.add_argument('-df',
                           help='Feature & class dataframe for ML, (example: '
                                'example_binary.txt) ',
                           required=True)

    # Optional
    inp_group = parser.add_argument_group(title='OPTIONAL INPUT')
    inp_group.add_argument('-df2',
                           help='Class data (if not in -df). Need to provide '
                                '-y_name',
                           default='')
    inp_group.add_argument('-sep', help='Deliminator', default='\t')
    inp_group.add_argument('-y_name',
                           help='Name of column in Y_file to predict',
                           default='Y')
    inp_group.add_argument('-test', help='File with testing lines', default='')
    inp_group.add_argument('-feat',
                           help='File with list of features (from x) to include',
                           default='all')
    inp_group.add_argument('-model', help='saved model', default='')
    inp_group.add_argument('-top', help='top # you want to show in the figures',
                           default=20)

    # Output arguments
    out_group = parser.add_argument_group(title='OUTPUT OPTIONS')
    out_group.add_argument('-save',
                           help='prefix for output files. CAUTION: will '
                                'overwrite!',
                           default='')

    # Default Hyper parameters
    params_group = parser.add_argument_group(title='DEFINE HYPERPARAMETERS')
    params_group.add_argument('-n_estimators', help='RF/GB parameter.',
                              default=100)
    params_group.add_argument('-max_depth',
                              help='RF/GB parameter. Grid Search [3, 5, 10]',
                              default=5)
    params_group.add_argument('-max_features',
                              help='RF/GB parameter. Grid Search [0.1, 0.25, '
                                   '0.5, 0.75, sqrt, log2, None]',
                              default='sqrt')
    params_group.add_argument('-n_jobs', '-p',
                              help='Number of processors for parallel '
                                   'computing (max for HPCC = 14)',
                              type=int, default=1)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    try:
        args.max_features = float(args.max_features)
    except:
        args.max_features = args.max_features

    args.n_estimators = int(args.n_estimators)
    args.max_depth = int(args.max_depth)

    ############################################
    ####### Load Dataframe & Pre-process #######
    ############################################

    df = dt.fread(args.df, header=True, sep=args.sep)
    df = df.to_pandas()
    df = df.set_index(df.columns[0], drop=True)

    # If features  and class info are in separate files, merge them:
    if args.df2 != '':
        start_dim = df.shape
        df_class = pd.read_csv(args.df2, sep=args.sep, index_col=0)
        df = pd.concat([df_class[args.y_name], df], axis=1, join='inner')
        print(
            'Merging the feature & class dataframes changed the dimensions '
            'from %s to %s (instance, features). '
            % (str(start_dim), str(df.shape)))

    # Specify Y column - default = Class
    if args.y_name != 'Y':
        df = df.rename(columns={args.y_name: 'Y'})

    # Filter out features not in feat file given - default: keep all
    if args.feat != 'all':
        print('Using subset of features from: %s' % args.feat)
        with open(args.feat) as f:
            features = f.read().strip().splitlines()
            features = ['Y'] + features
        df = df.loc[:, features]

    # Check for Nas
    if df.isnull().values.any() == True:
        if args.drop_na.lower() in ['t', 'true']:
            start_dim = df.shape
            df = df.dropna(axis=0)
            print(
                'Dropping rows with NA values changed the dimensions from %s '
                'to %s. '
                % (str(start_dim), str(df.shape)))
        else:
            print(df.columns[df.isnull().any()].tolist())
            print(
                'There are Na values in your dataframe.\n Impute them or add '
                '-drop_na True to remove rows with nas')
            quit()

    # Make sure Y is datatype numeric
    df['Y'] = pd.to_numeric(df['Y'], errors='raise')

    # Set up dataframe of test instances that the final models will be
    # applied to
    if args.test != '':
        df_all = df.copy()
        print('Removing test instances to apply model on later...')
        with open(args.test) as test_file:
            test_instances = test_file.read().splitlines()
        try:
            test_df = df.loc[test_instances, :]
            df = df.drop(test_instances)
        except:
            test_instances = [int(x) for x in test_instances]
            test_df = df.loc[test_instances, :]
            df = df.drop(test_instances)
    else:
        test_df = 'None'
        test_instances = 'None'

    X_train = df.drop(['Y'], axis=1)
    y_train = df['Y']

    X_test = test_df.drop(['Y'], axis=1)
    y_test = test_df['Y']

    if args.model != '':
        my_model = joblib.load(args.model)
    else:
        my_model = RandomForestClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=args.max_depth,
            max_features=args.max_features,
            criterion='gini',
            random_state=42,
            n_jobs=args.n_jobs)
        my_model.fit(X_train, y_train)

    # calculate and save the SHAP value matrix
    explainer = shap.Explainer(my_model)
    # prediction on test
    shap_values = explainer(X_train)
    plt.clf()
    shap.plots.beeswarm(shap_values, max_display=21, show=False)
    plt.savefig("SHAP_beeswarm_%s_training_top20.pdf" % args.save)
    values = pd.DataFrame(shap_values.values)
    values.index = X_train.index
    values.columns = X_train.columns
    # values.to_csv('SHAP_values_%s.txt'%args.save,sep='\t',header=True,index=True)

    # calculate the average absolute SHAP value of a column, and then rank
    # the features based on this average abs SHAP value
    values_abs = abs(values)
    values_abs_mean = values_abs.mean(axis=0)
    values_abs_mean_sorted = values_abs_mean.sort_values(ascending=False)
    values_abs_mean_sorted.to_csv(
        'SHAP_values_sorted_average_%s_training.txt' % args.save, sep='\t',
        header=True, index=True)

    # sort the column in the SHAP value matrix according to the average
    # absolute values
    values_sorted = values[values_abs_mean_sorted.index]
    values_sorted.to_csv('SHAP_values_sorted_%s_training.txt' % args.save,
                         sep='\t', header=True, index=True)

    # reorder the X_training, and remake the shape_values. Otherwise in the
    # following figure, the feature names would be wrong.
    X_train = X_train[values_abs_mean_sorted.index]
    explainer = shap.Explainer(my_model)
    shap_values = explainer(X_train)

    # get rid of the features beyond top # features
    values_top = values_sorted.iloc[:, 0:int(args.top)]
    Data = pd.DataFrame(shap_values.data)
    Data_top = Data.iloc[:, 0:int(args.top)]
    base_values_top = shap_values.base_values[0:int(args.top)]
    shap_values.values = np.array(values_top)
    shap_values.base_values = np.array(base_values_top)
    shap_values.data = np.array(Data_top)

    plt.clf()
    shap.plots.beeswarm(shap_values, max_display=int(args.top) + 1, show=False)
    plt.savefig("SHAP_beeswarm_%s_training_top%s_without_other_features.pdf" % (
        args.save, args.top))

    values_normalized = values_top.copy()
    for col in values_top.columns.to_list():
        if sum(abs(values_top[col])) != 0:
            # if col in values_abs_mean_sorted.index.to_list()[0:int(args.top)]:
            max_shap = np.max(values_top[col])
            min_shap = np.min(values_top[col])
            max_abs = np.max([abs(max_shap), abs(min_shap)])
            values_normalized[col] = values_top[col] / max_abs

    values_normalized_np = np.array(values_normalized)
    shap_values.values = values_normalized_np
    plt.clf()
    shap.plots.beeswarm(shap_values, max_display=int(args.top) + 1, show=False)
    plt.savefig("SHAP_beeswarm_%s_training_normalized_top%s.pdf" % (
        args.save, args.top))


if __name__ == '__main__':
    main()
