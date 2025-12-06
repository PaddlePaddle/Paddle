# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.io import dataloader
from paddle.io.dataloader import (
    BatchSampler as _BatchSampler,
    ChainDataset,
    ComposeDataset,
    ConcatDataset,
    Dataset,
    IterableDataset,
    RandomSampler as _RandomSampler,
    Sampler,
    SequenceSampler as SequentialSampler,
    Subset,
    SubsetRandomSampler as _SubsetRandomSampler,
    TensorDataset as _TensorDataset,
    WeightedRandomSampler as _WeightedRandomSampler,
    get_worker_info,
    random_split,
)
from paddle.io.dataloader.collate import default_collate_fn as default_collate
from paddle.io.reader import DataLoader as _DataLoader

from . import distributed


class BatchSampler(_BatchSampler):
    def __init__(
        self,
        sampler=None,
        batch_size=1,
        drop_last=False,
    ):
        super().__init__(
            sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )


class DataLoader(_DataLoader):
    def __init__(
        self,
        dataset=None,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device='',
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class RandomSampler(_RandomSampler):
    def __init__(
        self,
        data_source,
        replacement=False,
        num_samples=None,
        generator=None,
    ):
        super().__init__(
            data_source=data_source,
            replacement=replacement,
            num_samples=num_samples,
        )


class SubsetRandomSampler(_SubsetRandomSampler):
    def __init__(
        self,
        indices,
        generator=None,
    ):
        super().__init__(
            indices=indices,
        )


class TensorDataset(_TensorDataset):
    def __init__(self, *tensors):
        super().__init__(tensors)


class WeightedRandomSampler(_WeightedRandomSampler):
    def __init__(
        self,
        weights,
        num_samples=None,
        replacement=True,
        generator=None,
    ):
        super().__init__(weights, num_samples, replacement)


__all__ = [
    'Dataset',
    'IterableDataset',
    'TensorDataset',
    'ComposeDataset',
    'ChainDataset',
    'BatchSampler',
    'DataLoader',
    'get_worker_info',
    'Sampler',
    'SequentialSampler',
    'RandomSampler',
    'WeightedRandomSampler',
    'random_split',
    'Subset',
    'SubsetRandomSampler',
    'ConcatDataset',
    'dataloader',
    'default_collate',
    'distributed',
]
