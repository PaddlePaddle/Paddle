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

from __future__ import annotations

import bisect
import math
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Tuple,
    TypeVar,
)

from typing_extensions import Never, TypeVarTuple, Unpack

import paddle

from ... import framework

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator, Sequence

    from paddle import Tensor

_T = TypeVar('_T')
_Ts = TypeVarTuple('_Ts')


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
            "'{}' not implement in class "
            "{}".format('__getitem__', self.__class__.__name__)
        )

    def __len__(self) -> int:
        raise NotImplementedError(
            "'{}' not implement in class "
            "{}".format('__len__', self.__class__.__name__)
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
            "'{}' not implement in class "
            "{}".format('__iter__', self.__class__.__name__)
        )

    def __getitem__(self, idx: int) -> Never:
        raise RuntimeError(
            "'{}' should not be called for IterableDataset"
            "{}".format('__getitem__', self.__class__.__name__)
        )

    def __len__(self) -> Never:
        raise RuntimeError(
            "'{}' should not be called for IterableDataset"
            "{}".format('__len__', self.__class__.__name__)
        )


class TensorDataset(Dataset["Tensor"]):
    """
    Dataset defined by a list of tensors.

    Each tensor should be in shape of [N, ...], while N is the sample number,
    and each tensor contains a field of sample, :code:`TensorDataset` retrieve
    each sample by indexing tensors in the 1st dimension.

    Args:
        tensors(list|tuple): A list/tuple of tensors with same shape in the 1st dimension.

    Returns:
        Dataset: a Dataset instance wrapping tensors.

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> from paddle.io import TensorDataset


            >>> input_np = np.random.random([2, 3, 4]).astype('float32')
            >>> input = paddle.to_tensor(input_np)
            >>> label_np = np.random.random([2, 1]).astype('int32')
            >>> label = paddle.to_tensor(label_np)

            >>> dataset = TensorDataset([input, label])

            >>> for i in range(len(dataset)):
            ...     input, label = dataset[i]
            ...     # do something
    """

    tensors: Sequence[Tensor]

    def __init__(self, tensors: Sequence[Tensor]) -> None:
        if not framework.in_dynamic_mode():
            raise RuntimeError(
                "TensorDataset con only be used in imperative mode"
            )
        assert all(
            tensor.shape[0] == tensors[0].shape[0] for tensor in tensors
        ), "tensors not have same shape of the 1st dimension"
        self.tensors = tensors

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        return self.tensors[0].shape[0]


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class ComposeDataset(Dataset[Tuple[Unpack[_Ts]]]):
    """
    A Dataset which composes fields of multiple datasets.

    This dataset is used for composing fields of multiple map-style
    datasets of same length.

    Args:
        datasets(list of Dataset): List of datasets to be composed.

    Returns:
        Dataset: A Dataset which composes fields of multiple datasets.

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> from paddle.io import Dataset, ComposeDataset

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
            >>> dataset = ComposeDataset([RandomDataset(10), RandomDataset(10)])  # type: ignore[var-annotated]
            >>> for i in range(len(dataset)):
            ...     image1, label1, image2, label2 = dataset[i]
            ...     # do something
    """

    datasets: list[Dataset[Any]]

    def __init__(self, datasets: list[Dataset[Any]]) -> None:
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "input datasets should not be empty"
        for i, dataset in enumerate(self.datasets):
            assert isinstance(
                dataset, Dataset
            ), "each input dataset should be paddle.io.Dataset"
            assert not isinstance(
                dataset, IterableDataset
            ), "paddle.io.IterableDataset not supported"
            if i > 0:
                assert len(dataset) == len(
                    self.datasets[i - 1]
                ), "lengths of datasets should be same"

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx) -> tuple[Unpack[_Ts]]:
        sample = []
        for dataset in self.datasets:
            sample.extend(to_list(dataset[idx]))
        return tuple(sample)


class ChainDataset(IterableDataset[Any]):
    """
    A Dataset which chains multiple iterable-style datasets.

    This dataset is used for assembling multiple datasets which should
    be :ref:`api_paddle_io_IterableDataset`.

    Args:
        datasets(list of IterableDatasets): List of datasets to be chainned.

    Returns:
        paddle.io.IterableDataset: A Dataset which chains fields of multiple datasets.

    Examples:

        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> from paddle.io import IterableDataset, ChainDataset


            >>> # define a random dataset
            >>> class RandomDataset(IterableDataset):  # type: ignore[type-arg]
            ...     def __init__(self, num_samples):
            ...         self.num_samples = num_samples
            ...
            ...     def __iter__(self):
            ...         for i in range(10):
            ...             image = np.random.random([32]).astype('float32')
            ...             label = np.random.randint(0, 9, (1, )).astype('int64')
            ...             yield image, label
            ...
            >>> dataset = ChainDataset([RandomDataset(10), RandomDataset(10)])
            >>> for image, label in iter(dataset):
            ...     # do something
            ...     ...

    """

    def __init__(self, datasets: list[IterableDataset[Any]]):
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "input datasets should not be empty"
        for i, dataset in enumerate(self.datasets):
            assert isinstance(
                dataset, IterableDataset
            ), "ChainDataset only support paddle.io.IterableDataset"

    def __iter__(self) -> Iterator[Any]:
        for dataset in self.datasets:
            yield from dataset


class Subset(Dataset[_T]):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset.
        indices (sequence): Indices in the whole set selected for subset.

    Returns:
        List[Dataset]: A Dataset which is the subset of the original dataset.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> class RangeDataset(paddle.io.Dataset):  # type: ignore[type-arg]
            ...     def __init__(self, start, stop):
            ...         self.start = start
            ...         self.stop = stop
            ...
            ...     def __getitem__(self, index):
            ...         return index + self.start
            ...
            ...     def __len__(self):
            ...         return self.stop - self.start

            >>> # Example 1:
            >>> a = paddle.io.Subset(dataset=RangeDataset(1, 4), indices=[0, 2])
            >>> print(list(a))
            [1, 3]

            >>> # Example 2:
            >>> b = paddle.io.Subset(dataset=RangeDataset(1, 4), indices=[1, 1])
            >>> print(list(b))
            [2, 2]
    """

    dataset: Dataset[_T]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[_T], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int) -> _T:
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)


def random_split(
    dataset: Dataset[_T],
    lengths: Sequence[int],
    generator: Any | None = None,
) -> list[Subset[_T]]:
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator, optional): Generator used for the random permutation. Default is None then the DefaultGenerator is used in manual_seed().

    Returns:
        Datasets: A list of subset Datasets, which are the non-overlapping subsets of the original Dataset.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> paddle.seed(2023)
            >>> a_list = paddle.io.random_split(range(10), [3, 7])  # type: ignore[arg-type, var-annotated]
            >>> print(len(a_list))
            2

            >>> # output of the first subset
            >>> for idx, v in enumerate(a_list[0]):
            ...     print(idx, v) # doctest: +SKIP("The output depends on the environment.")
            0 7
            1 6
            2 5

            >>> # output of the second subset
            >>> for idx, v in enumerate(a_list[1]):
            ...     print(idx, v) # doctest: +SKIP("The output depends on the environment.")
            0 1
            1 9
            2 4
            3 2
            4 0
            5 3
            6 8
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(
                    f"Fraction at index {i} is not between 0 and 1"
                )
            n_items_in_split = int(math.floor(len(dataset) * frac))
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)

        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )
    # TODO(@Joejiong): support Variable or Tensor type with .tolist class member function.
    # For example var.item() and var.tolist()
    indices = paddle.randperm(sum(lengths)).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def _accumulate(
    iterable: Iterable[_T], fn: Callable[[_T, _T], _T] = lambda x, y: x + y
) -> Generator[_T, None, None]:
    """
    Return running totals

    Args:
        iterable: any iterable object for example dataset.
        y (x): one element in the iterable object.
        fn (x, y): Defaults to lambdax.

    Yields:
        yields total from beginning iterator to current iterator.

    Example code:

        .. code-block:: python

            >>> list(_accumulate([1, 2, 3, 4, 5]))
            [1, 3, 6, 10, 15]

            >>> import operator
            >>> list(_accumulate([1, 2, 3, 4, 5], operator.mul))
            [1, 2, 6, 24, 120]
    """

    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


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
        self.datasets = list(datasets)
        assert (
            len(self.datasets) > 0
        ), 'datasets should not be an empty iterable'
        for d in self.datasets:
            assert not isinstance(
                d, IterableDataset
            ), "ConcatDataset does not support IterableDataset"
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
