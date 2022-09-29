import pandas as pd
import numpy as np
from paddle.io import Dataset
from paddle.io import IterableDataset


def read_txt(path, **kwarg):
    text = pd.read_table(path, **kwarg)
    data = np.array(text).tolist()
    return data


class Decoding(object):
    def __init__(self, **kwarg):
        self.kwarg = kwarg    
    
    def __iter__(self):
        for (d, l) in self.data:
            yield d, l
    
    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        self.data = []
        for path in dataset_source:
            d = read_txt(path, **self.kwarg)
            self.data.extend(d)
        return self

