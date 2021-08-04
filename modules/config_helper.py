import yaml

def read_config(path):
    with open(path) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf