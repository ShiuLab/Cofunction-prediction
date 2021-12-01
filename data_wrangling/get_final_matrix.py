import pandas as pd

file1 = r"/pathways_matrix.csv" #file with labels for all gene pairs

## output of feature generation
## manuall change feature output to "GenePairs"
file2 = r"/pathways_matrix_feat_gen_input.csv.features_out" #output file from feature generation, without label


df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2, sep='\t')

df3 = pd.merge(df1, df2, on="Pair") #if files can't merge, "Pair" might not be present in columns name
                                    #manually change this in the file
#print(df3.head(5))

df3.to_csv(r"/pathways_input_matrix_model1.csv", index=False, sep='\t') #output entire matrix for MLpipeline input
