import numpy as np
import logging
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from modules.config_helper import read_config
from modules.evaluator import Evaluator
from modules.processor import DataReader


logging.basicConfig(
    format='%(asctime)s || %(levelname)s || %(module)s - %(funcName)s, line #%(lineno)d || %(message)s',
    level=logging.INFO,
    datefmt='%y/%b/%Y %H:%M:%S')


def cross_validate(conf, x_train, y_train):
    base_rf = RandomForestClassifier(random_state = conf['random_state'])

    max_features_opt = [x/10.0 for x in range(conf['max_features']['min'],
                                            10,
                                            conf['max_features']['interval'])]
    max_features_opt.extend(['auto','log2'])

    # many others to tune, but limiting for sake of compute power
    param_grid = { 
        'n_estimators': [int(x) for x in np.linspace(start = conf['estimators']['min'],
                                                    stop = conf['estimators']['max'],
                                                    num = conf['estimators']['num'])],
        'max_features': max_features_opt, 
        'class_weight': [{0:1,1:2}, {0:1, 1:1}, {0:1, 1:1.5}]
    }
    logging.info("Starting cross validation")
    CV_rfc = RandomizedSearchCV(estimator=base_rf, param_distributions=param_grid,
                                cv=conf['cv'])
    CV_rfc.fit(x_train, y_train)
    logging.info(f"Best parameters: {CV_rfc.best_params_}")

    best_model = CV_rfc.best_estimator_

    return best_model


def main(config_path, sample_size, y):

    conf = read_config(config_path)
    data_reader = DataReader('data/processed.pkl')

    data_reader.sample(size = sample_size)
    data_reader.define_y(y)
    x_train, x_test, y_train, y_test = data_reader.split_train_test()

    final_model = cross_validate(conf, x_train, y_train)
    final_model.fit(x_train, y_train)

    eval = Evaluator(final_model, x_test, y_test)
    eval.get_accuracy()
    eval.create_plots()
    eval.get_feat_imp()

    # TODO: roc, sensitivity at low alert rates
    # TODO: evaluate accuracy by unbalanced classes
    # TODO: upsample true class? something else?
    # TODO: formatting


if __name__ == "__main__":
    # TODO: validate arguments
    # TODO: set largest sample possible time-wise for final model
    # args = argparse.ArgumentParser()
    # args.add_argument('config_path',
    #                 default = 'model_config.yaml',
    #                 type = str)
    # args.add_argument('sample_size',
    #                 default = 0.1, 
    #                 type = float)
    # args.add_argument('y',
    #                 default = 'IsBadBuy',
    #                 type = str)

    # main(config_path = args.config_path,
    #      sample_size = args.sample_size,
    #      y = args.y)

    main('model_config.yaml', 0.01, 'IsBadBuy')