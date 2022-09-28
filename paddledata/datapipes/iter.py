import numpy as np
import more_itertools
from paddle.io import IterableDataset


class IterableMapper(IterableDataset):
    def __init__(self, map_func):
        self.map_func = map_func
    
    def __iter__(self):
        for d, l in self.dataset_source:
            data = self.map_func(d)
            label = l
            yield data, label

    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        return self


class IterableFilter(IterableDataset):
    def __init__(self, predicate):
        self.predicate = predicate

    def __iter__(self):
        for (d, l) in self.dataset_source:
            if self.predicate((d, l)):
                data = d
                label = l
                yield data, label

    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        return self


class IterableShuffler(IterableDataset):
    def __init__(self, replace=False):
        self.replace = replace
    
    def __iter__(self):
        n = more_itertools.ilen(self.dataset_source)
        data, label = [], []
        for d, l in self.dataset_source:
            data.append(d)
            label.append(l)
        for index in np.random.choice(np.arange(n), n, replace=self.replace).tolist():
            yield data[index], label[index]

    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        return self


class IterableBatcher(IterableDataset):
    def __init__(self, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        batch_data, batch_label = (), ()
        for d, l in self.dataset_source:
            batch_data = batch_data + (d,)
            batch_label = batch_label + (l,)
            if len(batch_label) == self.batch_size:
                yield batch_data, batch_label
                batch_data, batch_label = (), ()
        if not self.drop_last and len(batch_label) > 0:
            yield batch_data, batch_label

    """
    def __len__(self):
        num_samples = len(self.dataset_source)
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size
    """

    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        return self


class IterableDataPipes(object):
    def __init__(self, pipelines):
        self.pipelines = pipelines

    def __call__(self, data):
        for pipe in self.pipelines:
            try:
                data = pipe(data)
            except Exception as e:
                print("fail to perform processing [{}] with error: {}".format(pipe, e))
                raise e
        return data

