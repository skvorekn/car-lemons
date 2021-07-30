import os
import pandas as pd
import numpy as np
import logging
import datetime as dt

from pandas_profiling import ProfileReport

from sklearn.model_selection import train_test_split
from pandas_streaming.df import train_test_apart_stratify

from modules.utilities import split_x_y, log_prevalence

class DataReader():
    """
    Read and manipulate training data
    """

    __readers__ = {'.pkl': pd.read_pickle,
                   '.csv': pd.read_csv}

    __split_methods__ = {True: train_test_apart_stratify,
                         False: train_test_split}

    def __init__(self, path):
        self.path = path
        _, self.format = os.path.splitext(self.path)

        try:
            self.data = self.__readers__[self.format](self.path)
        except ValueError:
            raise ValueError(f"Invalid input file format '{self.format}'")

        logging.info(f"Read in data from {self.path}, size = {self.data.shape}")

    def drop_sparse_vars(self, threshold=0.5):
        missing_flag = self.data.isna().sum() > self.data.shape[0]*threshold
        drop_cols = list(missing_flag.index[missing_flag])
        self.data = self.data.drop(drop_cols, axis = 1)
        logging.info(f"Due to > {threshold} proportion missingness, dropped columns: {drop_cols}")

    def drop_ids(self, id_vars = ['RefId','WheelTypeID']):
        self.data.drop(id_vars, axis = 1)
        logging.info(f"Dropped ID variables: {id_vars}")

    def format_vars(self):
        self.data['PurchDate'] = self.data['PurchDate'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y').toordinal)

    def one_hot(self):
        self.data = pd.get_dummies(self.data).reset_index()
        mmrcols = self.data.filter(regex='MMR').columns.values
        self.data[mmrcols] = self.data[mmrcols].apply(lambda x: np.array(x, dtype='float64'))
        logging.info(f"Onehotted dataset shape: {self.data.shape}")

    # TODO - i.e. date parsing
    def create_derived_vars(self):
        pass

    # TODO: other methods to impute nulls
    def impute_nulls(self):
        self.data = self.data.fillna(self.data.median()).drop(['index'], axis = 1)
        logging.info("Nulls imputed with column median.")

    def sample(self, size, group=None):
        logging.info(f"Sampling data using sample size = {size}")
        if group is not None:
            selected_ids = self.data[group].drop_duplicates().sample(frac = size)
            self.data = self.data.merge(selected_ids, how = 'inner', on = group)
        else:
            self.data = self.data.sample(frac = size)
        logging.info(f"Sampled data size: {self.data.shape}")

    def create_profile_report(self, outpath = 'notebooks/Modeling Data Report.html'):
        profile = self.data.profile_report(title="Modeling Data Report")
        profile.to_file(outpath)
        logging.info(f"Profile report written to {outpath}")

    def define_y(self, dep_var):
        """
        Store dependent binary variable.
        """
        self.dep_var = dep_var
        log_prevalence(self.data[self.dep_var])

    def split_train_test(self, group=None, test_size=0.2):
        """Train test splilt stratified by dependent variable to handle class imbalance and, optionally, related observations.

        Args:
            group (str, optional): Name of group ID variable to which observations overlap. Defaults to None.
            test_size (float, optional): Fraction of data to use for testing. Defaults to 0.2.

        Returns:
            DataFrames:
        """
        self.test_size = test_size

        kwargs = {'test_size': self.test_size,
                  'random_state': 44133}
        if group_bool := (group is not None):
            kwargs['group'] = group
            kwargs['stratify'] = self.dep_var
        else:
            kwargs['stratify'] = self.y

        train, test = self.__split_methods__[group_bool](self.data, **kwargs)
        if group_bool:
            train = train.drop([group], axis = 1)
            test = test.drop([group], axis = 1)

        self.x_train, self.y_train = split_x_y(train, self.dep_var)
        self.x_test, self.y_test = split_x_y(test, self.dep_var)

        logging.info(f"Train data size: {self.x_train.shape}")
        log_prevalence(self.y_train)
        logging.info(f"Test data size: {self.x_test.shape}")
        log_prevalence(self.y_test)
        return self.x_train, self.x_test, self.y_train, self.y_test