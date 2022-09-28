import numpy as np
from paddle.io import Dataset


class MapMapper(Dataset):
    def __init__(self, map_func):
        self.map_func = map_func

    def __getitem__(self, idx):
        data = self.map_func(self.dataset_source[idx][0])
        label = self.dataset_source[idx][1]
        return data, label
    
    def __len__(self):
        return len(self.dataset_source)
    
    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        return self


class MapFilter(Dataset):
    def __init__(self, predicate):
        self.predicate = predicate
        
    def __getitem__(self, idx):
        data = self.dataset_source[self.filter_idx[idx]][0]
        label = self.dataset_source[self.filter_idx[idx]][1]
        return data, label
    
    def __len__(self):
        return len(self.filter_idx)

    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        filter_judge = [self.predicate(dataset_source[idx]) for idx in range(len(dataset_source))]
        self.filter_idx = np.where(filter_judge)[0]
        return self


class MapShuffler(Dataset):
    def __init__(self, replace=False):
        self.replace = replace

    def __getitem__(self, idx):
        n = len(self)
        r = np.random.choice(np.arange(n), n, replace=self.replace)
        data = self.x[r[idx]]
        label = self.y[r[idx]]
        return data, label
    
    def __len__(self):
        return len(self.x)
    
    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        self.x = [i[0] for i in dataset_source]
        self.y = [i[1] for i in dataset_source]
        """
        state = np.random.get_state()
        np.random.shuffle(self.x)
        np.random.set_state(state)
        np.random.shuffle(self.y)
        """
        return self


class MapBatcher(Dataset):
    def __init__(self, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __getitem__(self, idx):
        batch_data, batch_label = [], []
        for i in self.batch_indices[idx]:
            batch_data.append(self.dataset_source[i][0])
            batch_label.append(self.dataset_source[i][1])
        return batch_data, batch_label
    
    def __len__(self):
        return self.num_samples // self.batch_size
    
    def __call__(self, dataset_source):
        self.dataset_source = dataset_source
        self.num_samples = len(dataset_source)
        self.num_samples += int(not self.drop_last) * (self.batch_size - 1)
        self.batch_indices = []
        batch_idx = []
        for idx in range(len(dataset_source)):
            batch_idx.append(idx)
            if len(batch_idx) == self.batch_size:
                self.batch_indices.append(batch_idx)
                batch_idx = []
        if not self.drop_last and len(batch_idx) > 0:
            self.batch_indices.append(batch_idx)
        return self


class MapDataPipes(object):
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

