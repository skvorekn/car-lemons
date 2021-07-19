import logging
import datetime as dt
import os

def setup_logging(filename):
    model, _ = os.path.splitext(filename)

    start_time = dt.datetime.now()
    date = start_time.strftime("%Y-%b-%y")
    log_folder = f"logs/{date}"
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    logging.basicConfig(
        format='%(asctime)s || %(levelname)s || %(module)s - %(funcName)s, line #%(lineno)d || %(message)s',
        level=logging.INFO,
        datefmt='%y/%b/%Y %H:%M:%S',
        filename=f'{log_folder}/{model}_{start_time.strftime("%H:%M:%S")}.log')

def split_x_y(data, dep_var):
    y = data[dep_var]
    x = data.drop(dep_var, axis = 1)
    return x, y

def log_prevalence(y):
    logging.info(f"Prevalence: {100*round((y.value_counts()[1]/y.shape)[0],2)}%")
