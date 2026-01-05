# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import bisect
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)

from typing_extensions import Never

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

_T = TypeVar('_T')


class Dataset(Generic[_T]):
    """
    An abstract class to encapsulate methods and behaviors of datasets.

    All datasets in map-style(dataset samples can be get by a given key)
    should be a subclass of `paddle.io.Dataset`. All subclasses should
    implement following methods:

    :code:`__getitem__`: get sample from dataset with a given index. This
    method is required by reading dataset sample in :code:`paddle.io.DataLoader`.

    :code:`__len__`: return dataset sample number. This method is required
    by some implements of :code:`paddle.io.BatchSampler`

    see :code:`paddle.io.DataLoader`.

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> from paddle.io import Dataset

            >>> # define a random dataset
            >>> class RandomDataset(Dataset):  # type: ignore[type-arg]
            ...     def __init__(self, num_samples):
            ...         self.num_samples = num_samples
            ...
            ...     def __getitem__(self, idx):
            ...         image = np.random.random([784]).astype('float32')
            ...         label = np.random.randint(0, 9, (1, )).astype('int64')
            ...         return image, label
            ...
            ...     def __len__(self):
            ...         return self.num_samples
            ...
            >>> dataset = RandomDataset(10)
            >>> for i in range(len(dataset)):
            ...     image, label = dataset[i]
            ...     # do something
    """

    def __init__(self) -> None:
        pass

    def __getitem__(self, idx: int) -> _T:
        raise NotImplementedError(
            "'{}' not implement in class {}".format(
                '__getitem__', self.__class__.__name__
            )
        )

    def __len__(self) -> int:
        raise NotImplementedError(
            "'{}' not implement in class {}".format(
                '__len__', self.__class__.__name__
            )
        )

    if TYPE_CHECKING:
        # A virtual method for type checking only
        def __iter__(self) -> Iterator[_T]: ...


class IterableDataset(Dataset[_T]):
    """
    An abstract class to encapsulate methods and behaviors of iterable datasets.

    All datasets in iterable-style (can only get sample one by one sequentially, like
    a Python iterator) should be a subclass of :ref:`api_paddle_io_IterableDataset` . All subclasses should
    implement following methods:

    :code:`__iter__`: yield sample sequentially. This method is required by reading dataset sample in :ref:`api_paddle_io_DataLoader` .

    .. note::
        do not implement :code:`__getitem__` and :code:`__len__` in IterableDataset, should not be called either.

    see :ref:`api_paddle_io_DataLoader` .

    Examples:

        .. code-block:: python
            :name: code-example1

            >>> import numpy as np
            >>> from paddle.io import IterableDataset

            >>> # define a random dataset
            >>> class RandomDataset(IterableDataset): # type: ignore[type-arg]
            ...     def __init__(self, num_samples):
            ...         self.num_samples = num_samples
            ...
            ...     def __iter__(self):
            ...         for i in range(self.num_samples):
            ...             image = np.random.random([784]).astype('float32')
            ...             label = np.random.randint(0, 9, (1, )).astype('int64')
            ...             yield image, label
            ...
            >>> dataset = RandomDataset(10)
            >>> for img, label in dataset:
            ...     # do something
            ...     ...

    When :attr:`num_workers > 0`, each worker has a different copy of the dataset object and
    will yield whole dataset samples, which means samples in dataset will be repeated in
    :attr:`num_workers` times. If it is required for each sample to yield only once, there
    are two methods to configure different copy in each worker process to avoid duplicate data
    among workers as follows. In both the methods, worker information that can be getted in
    a worker process by `paddle.io.get_worker_info` will be needed.

    splitting data copy in each worker in :code:`__iter__`

        .. code-block:: python
            :name: code-example2

            >>> import math
            >>> import paddle
            >>> import numpy as np
            >>> from paddle.io import IterableDataset, DataLoader, get_worker_info

            >>> class SplitedIterableDataset(IterableDataset): # type: ignore[type-arg]
            ...     def __init__(self, start, end):
            ...         self.start = start
            ...         self.end = end
            ...
            ...     def __iter__(self):
            ...         worker_info = get_worker_info()
            ...         if worker_info is None:
            ...             iter_start = self.start
            ...             iter_end = self.end
            ...         else:
            ...             per_worker = int(
            ...                 math.ceil((self.end - self.start) / float(
            ...                     worker_info.num_workers)))
            ...             worker_id = worker_info.id
            ...             iter_start = self.start + worker_id * per_worker
            ...             iter_end = min(iter_start + per_worker, self.end)
            ...
            ...         for i in range(iter_start, iter_end):
            ...             yield np.array([i])
            ...
            >>> dataset = SplitedIterableDataset(start=2, end=9)
            >>> dataloader = DataLoader(
            ...     dataset,
            ...     num_workers=2,
            ...     batch_size=1,
            ...     drop_last=True)
            ...
            >>> for data in dataloader:
            ...     print(data) # doctest: +SKIP("The output depends on the environment.")
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[2]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[3]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[4]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[5]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[6]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[7]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[8]])

    splitting data copy in each worker by :code:`worker_init_fn`

        .. code-block:: python
            :name: code-example3

            >>> import math
            >>> import paddle
            >>> import numpy as np
            >>> from paddle.io import IterableDataset, DataLoader, get_worker_info

            >>> class RangeIterableDataset(IterableDataset): # type: ignore[type-arg]
            ...     def __init__(self, start, end):
            ...         self.start = start
            ...         self.end = end
            ...
            ...     def __iter__(self):
            ...         for i in range(self.start, self.end):
            ...             yield np.array([i])
            ...
            >>> dataset = RangeIterableDataset(start=2, end=9)

            >>> def worker_init_fn(worker_id):
            ...     worker_info = get_worker_info()
            ...
            ...     dataset: RangeIterableDataset = worker_info.dataset # type: ignore[assignment]
            ...     start = dataset.start
            ...     end = dataset.end
            ...     num_per_worker = int(
            ...         math.ceil((end - start) / float(worker_info.num_workers)))
            ...
            ...     worker_id = worker_info.id
            ...     dataset.start = start + worker_id * num_per_worker
            ...     dataset.end = min(dataset.start + num_per_worker, end)
            ...
            >>> dataloader = DataLoader(
            ...     dataset,
            ...     num_workers=2,
            ...     batch_size=1,
            ...     drop_last=True,
            ...     worker_init_fn=worker_init_fn)
            ...
            >>> for data in dataloader:
            ...     print(data) # doctest: +SKIP("The output depends on the environment.")
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[2]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[3]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[4]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[5]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[6]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[7]])
            Tensor(shape=[1, 1], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[8]])

    """

    def __init__(self) -> None:
        pass

    def __iter__(self) -> Iterator[_T]:
        raise NotImplementedError(
            "'{}' not implement in class {}".format(
                '__iter__', self.__class__.__name__
            )
        )

    def __getitem__(self, idx: int) -> Never:
        raise RuntimeError(
            "'{}' should not be called for IterableDataset{}".format(
                '__getitem__', self.__class__.__name__
            )
        )

    def __len__(self) -> Never:
        raise RuntimeError(
            "'{}' should not be called for IterableDataset{}".format(
                '__len__', self.__class__.__name__
            )
        )


class ConcatDataset(Dataset[_T]):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated

    Returns:
        Dataset: A Dataset which concatenated by multiple datasets.

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> from paddle.io import Dataset, ConcatDataset

            >>> # define a random dataset
            >>> class RandomDataset(Dataset):  # type: ignore[type-arg]
            ...     def __init__(self, num_samples):
            ...         self.num_samples = num_samples
            ...
            ...     def __getitem__(self, idx):
            ...         image = np.random.random([32]).astype('float32')
            ...         label = np.random.randint(0, 9, (1, )).astype('int64')
            ...         return image, label
            ...
            ...     def __len__(self):
            ...         return self.num_samples
            ...
            >>> dataset = ConcatDataset([RandomDataset(10), RandomDataset(10)])  # type: ignore[var-annotated]
            >>> for i in range(len(dataset)):
            ...     image, label = dataset[i]
            ...     # do something
    """

    @staticmethod
    def cumsum(sequence: Sequence[Any]) -> list[int]:
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset[Any]]) -> None:
        from paddle.io import IterableDataset as IoIterableDataset

        self.datasets = list(datasets)
        assert len(self.datasets) > 0, (
            'datasets should not be an empty iterable'
        )
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), (
                "ConcatDataset does not support IterableDataset"
            )
            assert not isinstance(d, IoIterableDataset), (
                "ConcatDataset does not support IterableDataset"
            )
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> _T:
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
