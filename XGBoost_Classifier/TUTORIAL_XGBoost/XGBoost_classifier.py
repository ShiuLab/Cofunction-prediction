"""
CREATED BY: Ally Schumacher
CODE ADOPTED FROM: https://shap-lrjball.readthedocs.io/en/docs_update/
example_notebooks/tree_explainer/Explaining%20the%20Loss%20of%20a%20Model.html
PURPOSE: Understanding how XGBoost is implemented to later be applied to
personal research.

NOTES: from https://machinelearningmastery.com/xgboost-loss-functions/
    - What is XGBoost?
        - It's implementation of gradient boosting ensemble algorithm.

"""
import shap
import sklearn
import xgboost
import numpy as np
import matplotlib.pylab as pl

###################################
### TRAIN AN XGBOOST CLASSIFIER ###
###################################
#
# # dataset from shap for example purposes
# X, y = shap.datasets.adult()
#
# # printing out what X and y are from this dataset
# # X is the features in dataset form and multiple columns with index #s
# # y is TRUE and FALSE values which would be for this binary classification
#
# print("X", '\n', X, '\n', "y", '\n', y)
#
# model = xgboost.XGBClassifier()
# model.fit(X, y)
#
# # compute the logistic log-loss
# model_loss = -np.log(model.predict_proba(X)[:, 1]) * y + -np.log(
#     model.predict_proba(X)[:, 0]) * (1 - y)
#
# # breaking down computing logistic log-loss
# # What is (X)[:,1]? -- this would be all rows for the 2nd column
# # What (X)[:, 0] -- this would be all rows for the 1st column
#
# print("Model loss", '\n', model_loss[:10])
#
# ############################################
# ### EXPLAIN LOG-LOSS WITH TREE EXPLAINER ###
# ############################################
#
# explainer = shap.TreeExplainer(model, X, feature_dependence="independent",
#                                model_output="logloss")
# print("loss log tree explainer", '\n', explainer.shap_values(X.iloc[:10, :],
#       y[:10]).sum(1) + np.array([explainer.expected_value(v) for v in y[:10]]))

################################################################################
#                           ADDITIONAL PRACTICE MODEL                          #
################################################################################

####################
### LOAD DATASET ###
####################
from sklearn.model_selection import train_test_split

X, y = shap.datasets.adult()
print("X", '\n', X, '\n', "y", '\n', y)
X_display, y_display = shap.datasets.adult(display=True)

# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=7)
# Understanding what each dataset looks likes
#training
print("X_train",'\n', X_train)
print("y_train",'\n', y_train)

#testing
print("X_test",'\n', X_test)
print("y_test",'\n', y_test)


# d_train = xgboost.DMatrix(X_train, label=y_train)
# d_test = xgboost.DMatrix(X_test, label=y_test)
#
# #######################
# ### TRAIN THE MODEL ###
# #######################
#
# params = {
#     "eta": 0.01,
#     "objective": "binary:logistic",
#     "subsample": 0.5,
#     "base_score": np.mean(y_train),
#     "eval_metric": "logloss"
# }
# model = xgboost.train(params, d_train, 5000, evals=[(d_test, "test")],
#                       verbose_eval=100, early_stopping_rounds=20)
#
# ###################################
# ### CLASSIFY FEATURE ATTRIBUTES ###
# ###################################
#
# xgboost.plot_importance(model)
# pl.title("xgboost.plot_importance(model)")
# pl.show()
#
# xgboost.plot_importance(model, importance_type="cover")
# pl.title('xgboost.plot_importance(model, importance_type="cover")')
# pl.show()
#
# xgboost.plot_importance(model, importance_type="gain")
# pl.title('xgboost.plot_importance(model, importance_type="gain")')
# pl.show()
#
# ###########################
# ### EXPLAIN PREDICTIONS ###
# ###########################
#
# # this takes a minute or two since we are explaining over 30 thousand samples
# # in a model with over a thousand trees
#
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)
#
# # Visualize many predictions
# shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_display.iloc[:1000,:])
#
# # Bar chart of mean importance
# shap.summary_plot(shap_values, X_display, plot_type="bar")
#
# # SHAP Summary Plot
# shap.summary_plot(shap_values, X)
