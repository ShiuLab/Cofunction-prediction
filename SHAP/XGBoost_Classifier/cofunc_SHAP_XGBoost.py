"""
CREATED BY: Ally Schumacher
DATE: 03/16/22

USAGE: python SHAP/XGBoost_Classifier/cofunc_SHAP_XGBoost.py
    -df SHAP/XGBoost_Classifier/3pathways_fitnessonly.csv
"""
import argparse
import os
from pathlib import Path
import datatable as dt
import shap
import sklearn
import xgboost
import numpy as np
import matplotlib.pylab as pl
from sklearn.model_selection import train_test_split
import joblib

#############################
### WHERE TO SAVE FIGURES ###
#############################

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "Allp_balanced_reordered_XGBoost"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="pdf", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        pl.tight_layout()
    pl.savefig(path, format=fig_extension, dpi=resolution)

##############################################
######### READ IN FILES WITH ARGPARSE ########
##############################################

# define arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('-df', '--data',
                       help='File with genepairs as index, labels, '
                            'and features', required=True)

args = argparser.parse_args()

# get file paths
file1 = Path(args.data)

####################
### LOAD DATASET ###
####################

dt_df = dt.fread(file1, sep='\t')
df = dt_df.to_pandas()
df = df.set_index(df.columns[0], drop=True)

print(df.head())
X = df.drop(['label'], axis=1)
y= df['label']
X_display = df.drop(['label'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=7)


# training
print("X_train",'\n', X_train)
print("X_train SHAPE",'\n', X_train.shape)

print("y_train",'\n', y_train)
print("y_train SHAPE",'\n', y_train.shape)

#testing
print("X_test",'\n', X_test)
print("X_test SHAPE",'\n', X_test.shape)

print("y_test",'\n', y_test)
print("y_test SHAPE",'\n', y_test.shape)



d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)

#######################
### TRAIN THE MODEL ###
#######################

params = {
    "eta": 0.01,
    "max_depth": 6,
    "subsample": 1,
    "objective": "binary:logistic",
    "base_score": np.mean(y_train),
    "feature_selector": "cyclic",
    "eval_metric": "auc"
}
model = xgboost.train(params, d_train, 100, evals=[(d_test, "test")],
                      verbose_eval=100, early_stopping_rounds=20)


joblib.dump(model, 'allp_xgboost_model.joblib')
###################################
### CLASSIFY FEATURE ATTRIBUTES ###
###################################

xgboost.plot_importance(model)
pl.title("xgboost.plot_importance(model)")
save_fig("xgboost.plot_importance(model)")
pl.show()

xgboost.plot_importance(model, importance_type="cover")
pl.title('xgboost.plot_importance(model, importance_type="cover")')
save_fig("xgboost.plot_importance_model_cover")
pl.show()

xgboost.plot_importance(model, importance_type="gain")
pl.title('xgboost.plot_importance(model, importance_type="gain")')
save_fig("xgboost.plot_importance_model_gain")
pl.show()

###########################
### EXPLAIN PREDICTIONS ###
###########################

# this takes a minute or two since we are explaining over 30 thousand samples
# in a model with over a thousand trees

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap_values_ind = shap.TreeExplainer(model).shap_values(X)


# Visualize many predictions
shap.force_plot(explainer.expected_value, shap_values[:10,:], X.iloc[:10,
                                                               :], show=False)
# SHAP Summary Plot
shap.summary_plot(shap_values, X_display, plot_type="dot",
                  alpha=0.8,show=False)
save_fig("dot_summary_SHAP_alpha08")

# shap.summary_plot(shap_values[:10,:], X_display[:10,:], plot_type="violin",
#                   alpha=0.8,show=False)
# save_fig("violin_summary_SHAP_alpha08")
#
# shap.summary_plot(shap_values[:10,:], X_display[:10,:], plot_type="bar",
#                   alpha=0.8,show=False)
# save_fig("bar_summary_SHAP_alpha08")

# Dependence plots
for name in X_train.columns:
    shap.dependence_plot(name, shap_values, X, display_features=X_display,
                         show=False)
save_fig("dependence_SHAP")

# Intereaction??
for name in X_train.columns:
    shap.dependence_plot(name, shap_values_ind, X, display_features=X_display,show=False)
save_fig("interaction_shap")
