import logging

def split_x_y(data, dep_var):
    y = data[dep_var]
    x = data.drop(dep_var, axis = 1)
    return x, y

def log_prevalence(y):
    logging.info(f"Prevalence: {100*round((y.value_counts()[1]/y.shape)[0],2)}%")
