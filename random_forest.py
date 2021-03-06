import numpy as np
import logging
import os
import ntpath

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
import joblib

from modules.config_helper import read_config
from modules.utilities import configure_script_args, setup_logging
from modules.evaluator import Evaluator
from modules.processor import DataReader
from modules.base_model import BaseModel

class RandomForest(BaseModel):
    def __init__(self, conf):
        self.conf = conf

    def process_data(self, path, y, id_group, sample_size=1):
        data_reader = DataReader(path)
        data_reader.create_derived_vars()
        data_reader.drop_sparse_vars()
        data_reader.drop_ids()
        data_reader.format_vars()
        data_reader.one_hot()
        data_reader.impute_nulls()
        data_reader.sample(size = sample_size, group = id_group)
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

    def group_generator(self, flatted_group, gkf):
        for train, _ in gkf.split(self.x_train, groups=flatted_group):
            yield np.unique(flatted_group[train], return_counts=True)[1]

    def cross_validate(self, group=None):
        logging.info("Setting up cross validation")
        base_rf = RandomForestClassifier(random_state = self.conf['random_state'])

        if group_bool := (group is not None):
            group_train = self.x_train[group].values
            cv = GroupKFold(n_splits=self.conf["cv"])
            self.x_train = self.x_train.drop([group], axis = 1)
            self.x_test = self.x_test.drop([group], axis = 1)
        else:
            cv = self.conf["cv"]

        self.generate_param_grid()

        logging.info("Starting cross validation")
        CV_rfc = RandomizedSearchCV(estimator=base_rf, 
                                    param_distributions=self.param_grid,
                                    cv=cv)
        if group_bool:
            CV_rfc.fit(self.x_train, self.y_train, groups=group_train)
        else:
            CV_rfc.fit(self.x_train, self.y_train)
        logging.info(f"Best parameters: {CV_rfc.best_params_}")

        self.best_model = CV_rfc.best_estimator_

        return self.best_model


def main(config_path, data_path, sample_size, y, group=None):

    setup_logging(ntpath.basename(__file__))
    conf = read_config(config_path)

    rf = RandomForest(conf)
    x_train, x_test, y_train, y_test = rf.process_data(data_path, y, group, sample_size)
    final_model = rf.cross_validate(group)
    final_model.fit(x_train, y_train)
    joblib.dump(final_model, "output/random_forest.sav")

    eval = Evaluator(final_model, x_test, y_test, os.path.join('output',__file__.split('/')[-1]))
    eval.get_accuracy()
    eval.create_plots()
    eval.get_feat_imp()

    # TODO: evaluate accuracy by unbalanced classes
    # TODO: upsample true class? something else?
    # TODO: CI formatting & run tests


if __name__ == "__main__":
    args = configure_script_args()

    main(args.config_path, args.input_path, args.sample_size, args.y, args.id_group)