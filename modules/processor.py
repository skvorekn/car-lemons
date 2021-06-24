import os
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from pandas_streaming.df import train_test_apart_stratify

from modules.utilities import split_x_y

class DataReader():
    """
    Read and manipulate training data
    """

    __readers__ = {'.pkl': pd.read_pickle,
                   '.csv': pd.read_csv}

    def __init__(self, path):
        self.path = path
        _, self.format = os.path.splitext(self.path)

        try:
            self.data = self.__readers__[self.format](self.path)
        except ValueError:
            raise ValueError(f"Invalid input file format '{self.format}'")

        logging.info(f"Read in data from {self.path}, size = {self.data.shape}")

    def sample(self, size):
        logging.info(f"Sampling data using sample size = {size}")
        self.data = self.data.sample(frac = size)
        logging.info(f"Sampled data size: {self.data.shape}")

    def define_y(self, dep_var):
        """
        Store dependent binary variable. Split into X and Y data.
        """
        self.dep_var = dep_var
        logging.info(f"Prevalence: {100*round((self.data[self.dep_var].value_counts()[1]/self.data.shape)[0],2)}%")

    def split_train_test(self, group=None, test_size=0.2):
        """Train test splilt stratified by dependent variable to handle class imbalance and, optionally, related observations.

        Args:
            group (str, optional): Name of group ID variable to which observations overlap. Defaults to None.
            test_size (float, optional): Fraction of data to use for testing. Defaults to 0.2.

        Returns:
            DataFrames:
        """
        self.test_size = test_size

        group_bool = group is not None
        split_type = {True: train_test_apart_stratify,
                      False: train_test_split}

        kwargs = {'test_size': self.test_size,
                  'random_state': 44133}
        if group_bool:
            kwargs['group'] = group
            kwargs['stratify'] = self.dep_var
        else:
            kwargs['stratify'] = self.y

        train, test = split_type[group_bool](self.data, **kwargs)

        self.x_train, self.y_train = split_x_y(train, self.dep_var)
        self.x_test, self.y_test = split_x_y(test, self.dep_var)

        logging.info(f"Train data size: {self.x_train.shape}")
        logging.info(f"Test data size: {self.x_test.shape}")
        return self.x_train, self.x_test, self.y_train, self.y_test