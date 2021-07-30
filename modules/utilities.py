import logging
import datetime as dt
import os
import argparse

def setup_logging(filename):
    model, _ = os.path.splitext(filename)

    start_time = dt.datetime.now()
    date = start_time.strftime("%Y-%b-%y")
    log_folder = os.path.join("logs", date)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    logging.basicConfig(
        format='%(asctime)s || %(levelname)s || %(module)s - %(funcName)s, line #%(lineno)d || %(message)s',
        level=logging.INFO,
        datefmt='%y/%b/%Y %H:%M:%S',
        filename=f'{log_folder}/{model}_{start_time.strftime("%H_%M_%S")}.log',
        filemode='w')

def configure_script_args():
    # TODO: validate arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path',
                    help = 'Location of config file containing model parameters',
                    default = 'model_config.yaml',
                    type = str)
    parser.add_argument('--input-path',
                    help = 'Location of raw training data',
                    default = 'data/training.csv',
                    type = str)
    parser.add_argument('--sample-size',
                    help = 'Proportion of data to use',
                    default = 0.1, 
                    type = float)
    parser.add_argument('--y',
                    help = 'Dependent variable name',
                    default = 'IsBadBuy',
                    type = str)
    parser.add_argument('--id-group',
                    help = 'Column identifying related observations, to prevent leakage in model training',
                    default = 'BYRNO',
                    type = str)
    args = parser.parse_args()
    return args

def split_x_y(data, dep_var):
    y = data[dep_var]
    x = data.drop(dep_var, axis = 1)
    return x, y

def log_prevalence(y):
    logging.info(f"Prevalence: {100*round((y.value_counts()[1]/y.shape)[0],2)}%")
