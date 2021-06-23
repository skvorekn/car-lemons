import os
import pandas as pd
import logging

from sklearn.model_selection import train_test_split

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
        Store dependent binary variable. Split into X and Y data.
        """
        self.dep_var = dep_var
        self.y = self.data[self.dep_var]
        self.x = self.data.drop([self.dep_var], axis = 1)
        logging.info(f"Prevalence: {100*round((self.y.value_counts()[1]/self.y.shape)[0],2)}%")

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