 import pandas as pd
import numpy as np
import datetime
import logging
from pprint import pprint
from matplotlib import pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import calibration_curve

logging.basicConfig(
    format='%(asctime)s || %(levelname)s || %(module)s - %(funcName)s, line #%(lineno)d || %(message)s',
    level=logging.INFO,
    datefmt='%y/%b/%Y %H:%M:%S')

# TODO: module
class DataReader():
    """
    Read and manipulate training data
    """
    def __init__(self, path):
        self.path = path
        self.format = os.path.splitext(self.path)[1]

        if self.format == '.pkl':
            self.data = pd.read_pickle(self.path)
        elif self.format == '.csv':
            self.data = pd.read_csv(self.path)
        else:
            raise ValueError("Invalid file format")
        logging.info(f"Read in data from {self.path}, size = {self.data.shape}")

    def sample(self, size):
        logging.info(f"Sampling data using sample size = {size}")
        self.data = self.data.sample(frac = size)
        logging.info(f"Sampled data size: {self.data.shape}")

    def define_y(self, dep_var):
        """
        Store dependent variable. Split into X and Y data.
        """
        self.dep_var = dep_var
        self.y = self.data[self.dep_var]
        self.x = self.data.drop([self.dep_var], axis = 1)

    # assuming observations are unrelated, otherwise there would be leakage between train/test
    # TODO: split by BYRNO
    def split_train_test(self, test_size=0.2):
        self.test_size = test_size
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, 
                                                            test_size = self.test_size,
                                                            random_state = 44133)
        logging.info(f"Train data size: {x_train.shape}")
        logging.info(f"Test data size: {x_test.shape}")
        return x_train, x_test, y_train, y_test


data_reader = DataReader('data/processed.pkl')
# TODO: set largest sample possible time-wise for final model
sample_size = 0.1
data_reader.sample(size = sample_size)
data_reader.define_y('IsBadBuy')
x_train, x_test, y_train, y_test = data_reader.split_train_test()

base_rf = RandomForestClassifier(random_state = 44133)
max_features_opt = [x/10.0 for x in range(1, 10, 2)]
max_features_opt.extend(['auto','log2'])
param_grid = { 
    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
    'max_features': max_features_opt,
    # 'criterion' :['gini', 'entropy'],
    # 'bootstrap': [True, False],
    # 'class_weight': [{0:1,1:2}, {0:0, 1:0}]
}
logging.info("Starting cross validation")
CV_rfc = RandomizedSearchCV(estimator=base_rf, param_distributions=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)
logging.info(f"Best parameters: {CV_rfc.best_params_}")

final_model = CV_rfc.best_estimator_
final_model.fit(x_train, y_train)
pred_class_probs = final_model.predict_proba(x_test)
true_index = list(final_model.classes_).index(1)
true_preds = [pred[true_index] for pred in pred_class_probs]

# TODO: module
def plot_calibration(y_test, true_preds):
    fig = plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_test, true_preds, n_bins=10)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-")
    ax2.hist(true_preds, range=(0, 1), bins=10,
                histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

plot_calibration(y_test, true_preds)
plt.savefig('output/test_calibration.png')

# TODO: roc, sensitivity at low alert rates
logging.info(f"Test set score: {final_model.score(x_test, y_test)}")
# TODO: check unbalanced class issues?

rf_feat_imp = pd.Series(final_model.feature_importances_, index=x_train.columns.values)
rf_feat_imp = rf_feat_imp.sort_values(ascending=False)
rf_feat_imp.to_csv('output/feature_importance.csv')
# VehBCost = acquisition cost paid for vehicle at time of purchase
# odometer reading
# auction/retail prices
# purchase date
# warranty cost