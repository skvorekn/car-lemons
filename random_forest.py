import numpy as np
import logging
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from modules.config_helper import read_config
from modules.utilities import configure_script_args, setup_logging
from modules.evaluator import Evaluator
from modules.processor import DataReader
from modules.base_model import BaseModel

# class RandomForest(BaseModel):
class RandomForest():
    def __init__(self, conf):
        self.conf = conf

    def process_data(self, path, y, id_group, sample_size=1):
        data_reader = DataReader(path)
        data_reader.drop_sparse_vars()
        data_reader.drop_ids()
        data_reader.format_vars()
        data_reader.one_hot()
        data_reader.impute_nulls()
        data_reader.sample(size = sample_size)
        data_reader.create_profile_report()
        data_reader.define_y(y)
        self.x_train, self.x_test, self.y_train, self.y_test = data_reader.split_train_test(group=id_group)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def generate_param_grid(self):
        max_features_opt = [x/10.0 for x in range(self.conf['max_features']['min'],
                                                10,
                                                self.conf['max_features']['interval'])]
        max_features_opt.extend(['auto','log2'])

        # many others to tune, but limiting for sake of compute power
        self.param_grid = { 
            'n_estimators': [int(x) for x in np.linspace(start = self.conf['estimators']['min'],
                                                        stop = self.conf['estimators']['max'],
                                                        num = self.conf['estimators']['num'])],
            'max_features': max_features_opt, 
            'class_weight': [{0:1,1:2}, {0:1, 1:1}, {0:1, 1:1.5}]
        }

        return self.param_grid

    def cross_validate(self):
        base_rf = RandomForestClassifier(random_state = self.conf['random_state'])

        self.generate_param_grid()

        logging.info("Starting cross validation")
        # TODO: also need to handle groups (BYRNO) in cross validation
        CV_rfc = RandomizedSearchCV(estimator=base_rf, param_distributions=self.param_grid,
                                    cv=self.conf['cv'])
        CV_rfc.fit(self.x_train, self.y_train)
        logging.info(f"Best parameters: {CV_rfc.best_params_}")

        self.best_model = CV_rfc.best_estimator_

        return self.best_model


def main(config_path, data_path, sample_size, y, group=None):

    setup_logging(__file__.split('/')[-1])
    conf = read_config(config_path)

    rf = RandomForest(conf)
    x_train, x_test, y_train, y_test = rf.process_data(data_path, y, group, sample_size)
    final_model = rf.cross_validate()
    final_model.fit(x_train, y_train)

    # TODO: clean these up
    eval = Evaluator(final_model, x_test, y_test, os.path.join('output',__file__.split('/')[-1]))
    eval.get_accuracy()
    eval.create_plots()
    eval.get_feat_imp()

    # TODO: roc, sensitivity at low alert rates
    # TODO: evaluate accuracy by unbalanced classes
    # TODO: upsample true class? something else?
    # TODO: formatting in CI
    # TODO: tests


if __name__ == "__main__":
    args = configure_script_args()

    # main(args.config_path, args.input_path, args.sample_size, args.y, args.id_group)
    main('model_config.yaml','data/training.csv',0.1,'IsBadBuy','BYRNO')