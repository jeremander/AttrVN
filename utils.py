import time
import pickle
import numpy as np
import pandas as pd

def time_format(seconds):
    """Formats a time into a convenient string."""
    s = ''
    if (seconds >= 3600):
        s += "%dh," % (seconds // 3600)
    if (seconds >= 60):
        s += "%dm," % ((seconds % 3600) // 60)
    s += "%.3fs" % (seconds % 60)
    return s

def timeit(func, verbose = True):
    """Decorator for annotating function call with elapsed time info."""
    def timed(*args, **kwargs):
        if verbose:
            start_time = time.time()
        result = func(*args, **kwargs)
        if verbose:
            elapsed_time = time.time() - start_time
            print(time_format(elapsed_time))
        return result
    return timed

def load_object(folder, obj_name, extension, verbose = True):
    """Load object with the given name from a file with the naming convention folder/obj_name.extension."""
    filename = folder + '/' + obj_name + '.' + extension
    did_load = False
    try:
        if verbose:
            print("\nLoading %s from '%s'..." % (obj_name, filename))
        if (extension == 'csv'):
            obj = pd.read_csv(filename)
            did_load = True
        elif (extension == 'pickle'):
            obj = pickle.load(open(filename, 'rb'))
            did_load = True
        if (did_load and verbose):
            print("Successfully loaded %s." % obj_name)
            return obj
    except:
        pass
    if (not did_load):
        raise IOError("Could not load %s from file." % obj_name)

def save_object(obj, folder, obj_name, extension, verbose = True):
    """Saves object to a file with naming convention folder/obj_name.extension. File format depends on the extension."""
    filename = folder + '/' + obj_name + '.' + extension
    did_save = False
    try:
        if verbose:
            print("\nSaving %s to '%s'..." % (obj_name, filename))
        if (extension == 'csv'):
            pd.DataFrame.to_csv(obj, filename, index = False)
            did_save = True
        elif (extension == 'pickle'):
            pickle.dump(obj, open(filename, 'wb'))
            did_save = True
        if (did_save and verbose):
            print("Successfully saved %s." % obj_name)
    except:
        pass
    if (not did_save):
        raise IOError("Failed to save %s to file." % obj_name)

def configparser_items_to_params(items):
    """Evaluate the found items (strings) from a ConfigParser. Strips away any commented material."""
    params = {}
    for (t, v) in items:    
        v = v.partition('#')[0].strip()   
        try:  # try to evaluate parameter
            params[t] = eval(v)
        except (NameError, SyntaxError):  # otherwise assume string
            params[t] = v
    return params   

def normalize(vec):
    """Normalizes a vector to have unit norm."""
    return vec / np.linalg.norm(vec)

def normalize_mat_rows(mat):
    """Normalizes the rows of a matrix, in-place."""
    for i in range(mat.shape[0]):
        mat[i] = normalize(mat[i])