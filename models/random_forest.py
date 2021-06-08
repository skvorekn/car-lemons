# high number of obs vs. feautures
# low bias, high variance method
# also could try knn or kernel svm

# also good interpretability vs. neural nets & svm (also qventus does not use nn)
# lasso is also highly interpretable but maybe not as accurate

# rf and kernel svm allow data to not be linear

# Purpose: predict if car purchased at an auction is a kick to provide best inventory selection to their customers

# check unbalanced class issues? 

import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import logging
from pprint import pprint

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class DataReader():
    def __init__(self, path):
        self.path = path
        self.data = pd.read_pickle(path)
        logging.info(f"Read in data from {self.path}")

    def get_xy(self, dep_var):
        self.y = self.data[dep_var]
        self.x = self.data.drop([dep_var], axis = 1)
        return self.x, self.y
 
# get a list of models to evaluate
def get_models():
	models = dict()
	# explore ratios from 10% to 100% in 10% increments
	for i in np.arange(0.1, 1.1, 0.1):
		key = '%.1f' % i
		# set max_samples=None to use 100%
		if i == 1.0:
			i = None
		models[key] = RandomForestClassifier(max_samples=i)
	return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    logging.info(datetime.datetime.now())
    logging.info(model.get_params())
	# define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


data_reader = DataReader('data/processed.pkl')
x_train, y_train = data_reader.get_xy('IsBadBuy')

models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	logging.info(f"Evaluating model {name}")
	scores = evaluate_model(model, x_train, y_train)
	# store the results
	results.append(scores)
	names.append(name)
	# summarize the performance along the way
	logging.info('Accuracy: %.3f (%.3f)' % (scores.mean(), scores.std()))

# from matplotlib import pyplot
# plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()

# final model
# evaluate on test set
# feature importance
