"""
CREATED BY: Ally Schumacher
PURPOSE: input data matrix with instances, labels, and features. Balance
dataset and then reorder fitness with corresponding gene to have the larger
fitness value of the gene pairs be in one column

NOTE:
    - instances = "genepairs"
    - y/label = "label"
    - SMF gene 1 = Fit1
    - SMF gene 2 = Fit2
    - DMF gene 1 & 2 = Fit12

DATE CREATED: 3/22/22
USAGE:
"""

#####################
### PRELIMINARIES ###
#####################

# Load libraries

import argparse
from pathlib import Path
import datatable as dt
import pandas as pd
from sklearn.utils import resample
import numpy as np

##############################################
######### READ IN FILES WITH ARGPARSE ########
##############################################

### DEFINE ARGUMENTS ###

parser = argparse.ArgumentParser()

# Info about input data

parser.add_argument(
    '-df', '--data',
    help='Feature & class dataframe. Must be specified',
    required=True)

parser.add_argument('-y_name', '--y_name', help='Name of label column in '
                                           'dataframe,' \
                                          ' default=Class', default='Class')

parser.add_argument('-sep', '--deliminator', help='Deliminator, default=","',
    default=',')

parser.add_argument('-save', '--Output',
                       help='output file for manipulated dataframe',
                       required=True)

args = parser.parse_args()

### GET FILE PATH ###
file1 = Path(args.data)

####################
### LOAD DATASET ###
####################

dt_df = dt.fread(file1, sep=',')
df = dt_df.to_pandas()

df = df.replace(['?', 'NA', 'na', 'n/a', '', '.'], np.nan)

print("Snap shot of input data", '\n', df.head())
print("Shape of input data ", '\n', df.shape)

### CATCH SITUATION WHERE "Y" IS INCORRECTLY SPECIFIED ###
try:
    df_classes = df[args.y_name]
except KeyError:
    print("\nERR: y_name is specified as %s: does not exist\n" % args.y_name)
    sys.exit(0)

# Specify class column - default = Class
if args.y_name != 'Class':
    df = df.rename(columns={args.y_name: 'label'})


###################
### DOWN SAMPLE ###
###################

### FIND MIN CLASS ###
min_size = (df.groupby('label').size()).min() - 1
print("smallest class is", min_size, "out of", df.shape)

### GET INDEXES FOR THE POSITIVE AND NEGATIVE INSTANCES ###

# NEGATIVE SET DATA FRAME FOR ONLY THE LABELS
df_neg = df.loc[df['label'] == 0]
print("NEG DF ONLY:", '\n', df_neg)

# POSITIVE SET DATA FRAME FOR ONLY THE LABELS
df_pos = df.loc[df['label'] == 1]
print("POS DF ONLY:", '\n', df_pos)

# DOWN SAMPLE THE MAJORITY CLASS
df_neg_resampled = resample(df_neg['label'],
                            replace=False,  # sample without replacement
                            n_samples=min_size,  # to match minority class
                            random_state=123)  # reproducible results

print("dfneg_resampled", '\n', df_neg_resampled)

### GET INDEXES ###

dfneg_index_list = list(df_neg_resampled.index.values)
print("neg index list", '\n', dfneg_index_list)

dfpos_index_list = list(df_pos.index.values)
print("pos index list", '\n', dfpos_index_list)

### GET NEW DATAFRAME FROM SELECTED INDEXES ###
df_pos_tomerge = df.loc[dfpos_index_list]
print("DF pos only", '\n', df_pos_tomerge)

df_neg_tomerge = df.loc[dfneg_index_list]
print("DF neg only", '\n', df_neg_tomerge)

print("Positive only data frame is:", df_pos_tomerge.shape, "and Negative "
                                                            "balanced subset "
                                                            "is",
      df_neg_tomerge.shape)

### MERGE DATA FRAMES INTO THE FINAL BALANCED MATRIX ###

dataframes = [df_pos_tomerge, df_neg_tomerge]
balanced_df = pd.concat(dataframes)

print("Balanced dataframe snap shot:", '\n', balanced_df.head())
print("Balanced dataframe shape", '\n', balanced_df.shape)
print("neg data frame shape was", df_neg_tomerge.shape, "+",
      df_pos_tomerge.shape, "which should equal when combined",
      balanced_df.shape)

######################################
### REORDER GENE PAIRS AND FITNESS ###
######################################

### SEPARATE GENE PAIRS ###

balanced_df[['gene1', 'gene2']] = df['genepairs'].str.split('_', expand=True)
print("added seperated pairs in new column", '\n', balanced_df.head())

### RE-ORDER
balanced_df = balanced_df.drop(columns='genepairs')

balanced_df = balanced_df.filter(['gene1','gene2','label', 'Fit1','Fit2',
                                 'Fit12'], axis=1)

print("after dropping and rearranging column order", '\n', balanced_df.head())


### FIND THE BIGGEST FITNESS VALUE BETWEEN THE PAIR

df_g1bigger = balanced_df.loc[balanced_df['Fit1'] > balanced_df['Fit2']]
df_g2bigger = balanced_df.loc[balanced_df['Fit2'] > balanced_df['Fit1']]

print("Gene 1 bigger snapshot", '\n', df_g1bigger.head())
print("Gene 1 bigger shape", '\n', df_g1bigger.shape)

print("Gene 2 bigger snapshot", '\n', df_g2bigger.head())
print("Gene 2 bigger shape", '\n', df_g2bigger.shape)


### REORDER AND SWITCH COLUMN NAMES ACCORDINGLY ###

df_g2bigger = df_g2bigger[['gene2', 'gene1', 'label', 'Fit2', 'Fit1', 'Fit12']]
print("Gene 2 after reorder", '\n', df_g2bigger.head())
print("Gene 2 after reorder shape", '\n', df_g2bigger.shape)

df_g2bigger = df_g2bigger.rename(columns={'gene2': 'gene1', 'gene1': 'gene2',
                                          'Fit2': 'Fit1', 'Fit1': 'Fit2'})
print("Gene 2 after rename", '\n', df_g2bigger.head())
print("Gene 2 after rename", '\n', df_g2bigger.shape)


##############
### OUTPUT ###
##############

dataframes = [df_g2bigger, df_g1bigger]
merged = pd.concat(dataframes, axis=0).reset_index(drop=True)
print("merged", '\n', merged.head())
print("merged", '\n', merged.shape)

merged['GenePairs'] = merged['gene1'].str.cat(merged['gene2'], sep="_")
merged = merged.filter(['GenePairs', 'label', 'Fit1', 'Fit2', 'Fit12'], axis=1)
print("final merge", '\n', merged.head())

merged.to_csv(args.Output, index=False)

print("DONE, YAY!!!!!")
