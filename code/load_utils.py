import pickle as pkl

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield list(map(int,iterable[ndx:min(ndx + n, l)]))