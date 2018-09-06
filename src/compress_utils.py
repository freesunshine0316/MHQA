import cPickle
import gzip

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object

def save(obj, outpath):
    if outpath.endswith('.gz'):
        save_zipped_pickle(obj, outpath, protocol=2)
    else:
        with open(outpath, 'wb') as outfile:
            cPickle.dump(obj, outfile, protocol=2)

def load(inpath):
    if inpath.endswith('.gz'):
        obj = load_zipped_pickle(inpath)
    else:
        with open(inpath, "rb") as f:
            obj = cPickle.load(f)
    return obj
