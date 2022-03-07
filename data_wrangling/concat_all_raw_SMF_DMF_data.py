# This script is used to get,filter,and produce a file that contains all fitness data for all gene pairs
# data of single and double mutant gene pairs.
# This data was gathered by Costanzo 2016 https://doi.org/10.1126/science.aaf1420
# Data downloaded from: https://datadryad.org/stash/dataset/doi:10.5061/dryad.4291s
# Usage notes: Data File S1. Raw genetic interaction datasets - Pair-wise interaction format
# Files: SGA_ExE.txt, SGA_NxN.txt, SGA_ExN_NxE.txt

import pandas as pd

E_E = pd.read_table("/SGA_ExE.txt", sep='\t')
N_N = pd.read_table("/SGA_NxN.txt", sep='\t')
E_N = pd.read_table("/SGA_ExN_NxE.txt", sep='\t')

# E_E.head()
# E_E.isnull().sum(axis=0)
# N_N.head()
# N_N.isnull().sum(axis=0)
# E_N.head()

# print('Shape of data (rows, cols):', '\n', E_E.shape)
# print('\nSnapshot of data:', '\n', E_E.iloc[:6, :])

frames = [E_E, N_N, E_N]
all_fitness_data = pd.concat(frames)
all_fitness_data.columns = [c.replace(' ', '_') for c in all_fitness_data.columns]
# all_fitness_data.head()

# all_fitness_data.info() #need columns 'Query_allele_name', 'Array_allele_name', 'Query_single_mutant_fitness_(SMF)', 'Array_SMF', 'Double_mutant_fitness'
df = all_fitness_data.iloc[:, [1, 3, 7, 8, 9]]

df['Query_allele_name'] = df['Query_allele_name'].str.split('-').str[0]
df['Array_allele_name'] = df['Array_allele_name'].str.split('-').str[0]
# df.head()

# df.isnull().sum(axis=0)
# df.info()
new_df = df.dropna()
# new_df.info()

new_df.to_csv("/raw_filtered_genepairs_SMF_DMF.csv", index = False)
# new_data = pd.read_csv("/raw_filtered_genepairs_SMF_DMF.csv, sep=',')
# new_data.head()