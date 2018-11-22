import pickle as pkl

"""
This module defines helper functions to load, save and batch data.
"""

def save_obj(obj, name):
    """ 
    Shortcut function to save an object as pkl
    Args:
        obj: object to save
        name: filename of the object
    """
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(name):
    """ 
    Shortcut function to load an object from pkl file
    Args:
        name: filename of the object
    Returns:
        obj: object to load
    """
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

def batch(iterable, n=1):
    """ Batch iterator creator.
    Args:
        iterable: iterable to batch
        n: batch size
    Returns:
        Iterator over batches.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield list(map(int,iterable[ndx:min(ndx + n, l)]))