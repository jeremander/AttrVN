import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Tuple, TypeVar, Union

F = TypeVar('F', bound = Callable[..., Any])
AnyPath = Union[str, Path]


def time_format(seconds: float) -> str:
    """Formats a time into a convenient string."""
    s = ''
    if (seconds >= 3600):
        s += "%dh," % (seconds // 3600)
    if (seconds >= 60):
        s += "%dm," % ((seconds % 3600) // 60)
    s += "%.3fs" % (seconds % 60)
    return s

def timeit(func: F, verbose: bool = True) -> F:
    """Decorator for annotating function call with elapsed time info."""
    def timed(*args: Any, **kwargs: Any) -> Any:
        if verbose:
            start_time = time.time()
        result = func(*args, **kwargs)
        if verbose:
            elapsed_time = time.time() - start_time
            print(time_format(elapsed_time))
        return result
    return timed  # type: ignore

def name_and_extension(path: AnyPath) -> Tuple[str, str]:
    """Given a path with the naming convention obj_name.extension, returns (obj_name, extension)."""
    path = Path(path)
    toks = path.name.split('.')
    return ('.'.join(toks[:-1]), toks[-1])

def load_object(path: AnyPath, verbose: bool = True) -> Any:
    """Load object from a file with the naming convention obj_name.extension."""
    (obj_name, extension) = name_and_extension(path)
    did_load = False
    if verbose:
        print(f'Loading {obj_name} from {path}')
    if (extension == 'csv'):
        obj = pd.read_csv(path)
        did_load = True
    elif (extension == 'pickle'):
        obj = pickle.load(open(path, 'rb'))
        did_load = True
    if (did_load and verbose):
        print(f'Successfully loaded {path}')
        return obj
    if (not did_load):
        raise IOError(f'Could not load {path}')

def save_object(obj: Any, path: AnyPath, verbose: bool = True) -> None:
    """Saves object to a file with naming convention folder/obj_name.extension. File format depends on the extension."""
    (obj_name, extension) = name_and_extension(path)
    did_save = False
    if verbose:
        print(f'Saving {obj_name} to {path}')
    if (extension == 'csv'):
        pd.DataFrame.to_csv(obj, path, index = False)
        did_save = True
    elif (extension == 'pickle'):
        pickle.dump(obj, open(path, 'wb'))
        did_save = True
    if (did_save and verbose):
        print(f'Successfully saved {path}')
    if (not did_save):
        raise IOError(f'Failed to save {path}')

def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalizes a vector to have unit norm."""
    return vec / np.linalg.norm(vec)

def normalize_mat_rows(mat: np.ndarray) -> None:
    """Normalizes the rows of a matrix, in-place."""
    for i in range(mat.shape[0]):
        mat[i] = normalize(mat[i])