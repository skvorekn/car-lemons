from abc import ABCMeta, abstractclassmethod

class BaseModel(ABCMeta):
    def __init__(self):
        pass

    @abstractclassmethod
    def process_data(self, path):
        """
        Use methods from DataReader to process data for modeling
        """
        pass

    @abstractclassmethod
    def generate_param_grid(self):
        """
        Returns:
            dict{param: [options]} of parameters to try in cross validation
        """
        pass

    @abstractclassmethod
    def cross_validate(self):
        """
        Returns:
            model: best estimator after cross validation
        """
        # TODO: automated hyperparameter tuning, like gradient descent, 
        # bayesian optimization (Hyperopt library), or evolutionary algorithms
        pass