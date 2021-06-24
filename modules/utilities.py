def split_x_y(data, dep_var):
    y = data[dep_var]
    x = data.drop(dep_var, axis = 1)
    return x, y
