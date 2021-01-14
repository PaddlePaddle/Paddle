#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class _DatasetFetcher(object):
    def __init__(self, dataset, auto_collate_batch, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collate_batch = auto_collate_batch
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, batch_indices):
        raise NotImplementedError("'fetch' not implement for class {}".format(
            self.__class__.__name__))


class _IterableDatasetFetcher(_DatasetFetcher):
    def __init__(self, dataset, auto_collate_batch, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collate_batch,
                                              collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def fetch(self, batch_indices):

        if self.auto_collate_batch:
            data = []
            for _ in batch_indices:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    break
            if len(data) == 0 or (self.drop_last and
                                  len(data) < len(batch_indices)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)

        if self.collate_fn:
            data = self.collate_fn(data)
        return data


class _MapDatasetFetcher(_DatasetFetcher):
    def __init__(self, dataset, auto_collate_batch, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collate_batch, collate_fn, drop_last)

    def fetch(self, batch_indices):
        if self.auto_collate_batch:
            data = [self.dataset[idx] for idx in batch_indices]
        else:
            data = self.dataset[batch_indices]

        if self.collate_fn:
            data = self.collate_fn(data)
        return data
