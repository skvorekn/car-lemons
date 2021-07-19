from abc import ABCMeta, abstractclassmethod, abstractmethod

class BaseModel(ABCMeta):
    def __init__(self, conf):
        self.conf = conf

    @abstractmethod
    def process_data(self):
        """
        Use methods from DataReader to process data for modeling
        """
        pass

    @abstractmethod
    def generate_param_grid(self):
        """
        Returns:
            dict{param: [options]} of parameters to try in cross validation
        """
        pass

    @abstractmethod
    def cross_validate(self):
        """
        Returns:
            model: best estimator after cross validation
        """
        # TODO: automated hyperparameter tuning, like gradient descent, 
        # bayesian optimization (Hyperopt library), or evolutionary algorithms
        pass