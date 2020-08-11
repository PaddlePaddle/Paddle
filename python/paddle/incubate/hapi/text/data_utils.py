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

import copy
import collections
import functools
import inspect
import io
import json
import math
import os
import six
import unicodedata

import numpy as np
import paddle
from paddle.dataset.common import DATA_HOME
from paddle.io import Dataset
from paddle.incubate.hapi.download import get_path_from_url
from paddle.fluid.dygraph.parallel import ParallelEnv

from paddle.incubate.hapi.text.utils import InitTrackerMeta

# from .utils import InitTrackerMeta
# __all__ = ['Stack', 'Pad', 'Tuple']  # batchify
# __all__ += ['']  # dataset helper, sampler helper
# __all__ += ['']  # transform


class Stack(object):
    """
    Stack the input data samples to construct the batch.
    The N input samples must have the same shape/length and will be stacked to construct a batch.

    Args:
        axis (int, optional): The axis in the result data along which the input data are stacked. Default: 0.
        dtype (str|numpy.dtype, optional): The value type of the output. If it is set to None, the input data type is used. Default: None.

    Example:
        .. code-block:: python

            from paddle.incubate.hapi.text.data_utils import Stack
            # Stack multiple lists
            a = [1, 2, 3, 4]
            b = [4, 5, 6, 8]
            c = [8, 9, 1, 2]
            Stack()([a, b, c])
            '''
            [[1 2 3 4]
             [4 5 6 8]
             [8 9 1 2]]
             '''
    """

    def __init__(self, axis=0, dtype=None):
        self._dtype = dtype

    def __call__(self, data):
        """
        Batchify the input data

        Args:
            data (list): The input data samples.
        Returns:
            numpy.ndarray: Batch data.

        """
        data = np.stack(data).astype(self._dtype) if self._dtype else np.stack(
            data)
        return data


class Pad(object):
    """
    Return a callable that pads and stacks data.

    Args:
        pad_val (float|int, optional): The padding value. Default: 0.
        axis (int, optional): The axis to pad the arrays. The arrays will be padded to the largest dimension at axis. For example, 
            assume the input arrays have shape (10, 8, 5), (6, 8, 5), (3, 8, 5) and the axis is 0.Each input will be padded into 
            (10, 8, 5) and then stacked to form the final output, which has shape（3, 10, 8, 5). Default: 0.
        ret_length (int, optional): Length of the output. Default: None.
        dtype (str|numpy.dtype, optional): The value type of the output. If it is set to None, the input data type is used. Default: None.

    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.data_utils import Pad
            # Inputs are multiple lists
            a = [1, 2, 3, 4]
            b = [4, 5, 6]
            c = [8, 2]
            Pad(pad_val=0)([a, b, c])
            '''
            [[1. 2. 3. 4.]
                [4. 5. 6. 0.]
                [8. 2. 0. 0.]]
            '''
     """

    def __init__(self, pad_val=0, axis=0, ret_length=None, dtype=None):
        self._pad_val = pad_val
        self._axis = axis
        self._ret_length = ret_length
        self._dtype = dtype

    def __call__(self, data):
        """
        Batchify the input data.
        The input can be list of numpy.ndarray, list of numbers. 

        The arrays will be padded to the largest dimension at axis and then stacked to form the final output. 
        In addition, the function will output the original dimensions at the axis if ret_length is not None.

        Args:
            data (list(numpy.ndarray)|list(list)): List of samples to pad and stack.

        Returns:
            numpy.ndarray: Batch data, data in the minibatch. Shape is (N, …)
            numpy.ndarray (optional): Includes original length and output length. This will only be returned in ret_length is not None.
        """
        arrs = [np.asarray(ele) for ele in data]
        original_length = [ele.shape[self._axis] for ele in arrs]
        max_size = max(original_length)
        ret_shape = list(arrs[0].shape)
        ret_shape[self._axis] = max_size
        ret_shape = (len(arrs), ) + tuple(ret_shape)
        ret = np.full(
            shape=ret_shape,
            fill_value=self._pad_val,
            dtype=arrs[0].dtype if self._dtype is None else self._dtype)
        for i, arr in enumerate(arrs):
            if arr.shape[self._axis] == max_size:
                ret[i] = arr
            else:
                slices = [slice(None) for _ in range(arr.ndim)]
                slices[self._axis] = slice(0, arr.shape[self._axis])
                if slices[self._axis].start != slices[self._axis].stop:
                    slices = [slice(i, i + 1)] + slices
                    ret[tuple(slices)] = arr
        if self._ret_length is None:
            return ret
        else:
            return ret, np.asarray(original_length, self._ret_length)


class Tuple(object):
    """
    Wrap multiple batchify functions together. The input functions will be applied to the corresponding input fields.
    Each data sample should be a list or tuple containing multiple attributes. The i'th batchify function stored in 
    Tuple will be applied on the i'th attribute. For example, each data sample is (nd_data, label). 

    Args:
        fn (list|tuple|callable): The batchify functions to wrap.
        *args (tuple of callable): The additional batchify functions to wrap.

    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.data_utils import Tuple, Pad, Stack
            batchify_fn = Tuple(Pad(axis=0, pad_val=0), Stack())
    """

    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn, ) + args
        for i, ele_fn in enumerate(self._fn):
            assert callable(
                ele_fn
            ), 'Batchify functions must be callable! type(fn[%d]) = %s' % (
                i, str(type(ele_fn)))

    def __call__(self, data):
        """
        Batchify the input data.

        Args:
            data (list): The samples to batchfy. Each sample should contain N attributes.
        Returns:
            tuple: A tuple of length N. Contains the batchified result of each attribute in the input.
        """

        assert len(data[0]) == len(self._fn),\
            'The number of attributes in each data sample should contain' \
            ' {} elements'.format(len(self._fn))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
        return tuple(ret)


class SimpleDataset(Dataset):
    """
    Decorates dataset with shuffle, sort and other transformations.
    It acts as some specific sampler or iterator for dataset.
    Args:
        dataset (list|Dataset): A dataset-like object. It can be a list or a subclass of Dataset.
    """

    def __init__(self, data):
        self.data = data
        self._transform_func = None

    def __iter__(self):
        for idx in range(len(self.data)):
            yield self.data[idx]

    def __getitem__(self, idx):
        return self._transform_func(self.data[
            idx]) if self._transform_func else self.data[idx]

    def __len__(self):
        return len(self.data)

    def shuffle(self, buffer_size=-1, seed=None):
        """
        Shuffle the dataset according to the given buffer size and random seed.

        Args:
            buffer_size (int): Buffer size for shuffle. if buffer_size < 0, buffer_size is the length of the dataset. Default: -1. 
            seed (int): Seed for the random. Default: None.
        Returns:
            SimpleDataset: A new object.
        """
        if seed is not None:
            np_rand_state_bak = np.random.get_state()
            np.random.seed(seed)
        buffer_size = len(self.data) if buffer_size < 0 else buffer_size
        shuffled_data = []
        for i in range(0, len(self.data), buffer_size):
            buf = list(range(i, i + buffer_size))
            np.random.shuffle(buf)
            shuffled_data.extend([self.data[idx] for idx in buf])
        if seed is not None:
            np.random.set_state(np_rand_state_bak)
        return SimpleDataset(shuffled_data)

    def sort(self, cmp=None, key=None, reverse=False, buffer_size=-1):
        """
        Sort the dataset according to the given callable cmp or key.
        Args:
            cmp (callable): The function of comparison. Default: None. 
            key (callable): Return elements to be compared. Default: None.
            reverse (bool): If True, it means sorted results are in descending order, and False means in ascending order. 
                Default: False.
            buffer_size (int): Buffer size for sort. If buffer_size < 0 or buffer_size is more than the length of the data, 
                buffer_size will be set to the length of the data. Default: -1.
        Returns:
            SimpleDataset: A new object.
        """
        sorted_data = []
        buffer_size = len(self.data) if buffer_size < 0 else buffer_size
        if key:
            key = lambda x, y: cmp(self.data[x], self.data[y])
        elif cmp:
            key = functools.cmp_to_key(
                lambda x, y: cmp(self.data[x], self.data[y]))
        for i in range(0, len(self.data), buffer_size):
            sorted_data.extend(
                sorted(
                    range(i, i + buffer_size), key=key, reverse=reverse))
        return self

    def filter(self, predicate_func):
        """
        Filter the dataset with predicate_func.

        Args:
            predicate_func (callable): Return whether the data can be left in dataset.
        Returns:
            SimpleDataset: A new object.
        """
        filted_data = [
            self.data[idx] for idx in range(len(self.data))
            if predicate_func(self.data[idx])
        ]
        return SimpleDataset(filted_data)

    def apply(self, transform_func, lazy=False):
        """
        Transformations would be performed to dataset. It includes `Shuffle`, `sort`, `fit` and `shard`.

        Args:
            transform_func (callable): Transformations to be performed. It receives single
                sample as argument rather than dataset.
            lazy (bool): If True, transformations would be delayed and performed when calling
                `__getitem__`, which would run multi-times for each sample but can take
                advantage of DataLoader multi-processing. Defalt: False.
        Returns:
            SimpleDataset: A new object.
        """
        if lazy:
            self._transform_func = transform_func
        else:
            applied_data = [
                transform_func(self.data[idx])
                for idx in range(len(self.data))
            ]
            return SimpleDataset(applied_data)
        return self

    def shard(self, num_replicas=None, rank=None):
        """
        Operates slice using multi GPU.
        Args:
            num_replicas (int, optional): The number of training process, and is also the number of GPU cards used in training. 
                Default: None.
            rank (int, optional): Number of training process. Equals to the value of the environment variable PADDLE_TRAINER_ID.
                Default: None.
        Returns:
            SimpleDataset: A new object.
        """
        if num_replicas is None:
            num_replicas = ParallelEnv().nranks
        if rank is None:
            rank = ParallelEnv().local_rank
        num_samples = int(math.ceil(len(self.data) * 1.0 / num_replicas))
        total_size = num_samples * num_replicas
        # add extra samples to make it evenly divisible
        sharded_data = [
            self.data[idx]
            for idx in list(range(len(self.data))) + list(
                range(total_size - len(self.data)))
            if idx % num_replicas == rank
        ]
        return SimpleDataset(sharded_data)

    # def __getattribute__(self, name):
    #     return super().__getattribute__(name)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class SamplerHelper(object):
    """
    Sampler Factory. chain of sampling strategies

    Every SamplerHelper subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    Also can be used as batch iterator instead of indices iterator when `iterator`
    yield samples rather than indices by initializing `iterator` with a iterable
    dataset.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`DataLoader`, but is expected in any
              calculation involving the length of a :class:`DataLoader`.
    Args:
        dataset (collections.Iterable|Dataset): Input dataset for SamplerHelper.
        iterable (collections.Iterable|callable, optional): Iterator fof dataset. Default: None.
    """

    # chain sampler
    def __init__(self, dataset, iterable=None):
        self.data_source = dataset
        self.iterable = iterable
        if isinstance(dataset, collections.Iterable) and iterable is None:
            # iterable-style datasets
            self.iterable = dataset

    def __iter__(self):
        if self.iterable is None:
            return iter(range(len(self.data_source)))
        elif isinstance(self.iterable, collections.Iterable):
            return iter(self.iterable)
        elif callable(self.iterable):
            return self.iterable()
        else:
            raise ValueError(
                "`iterable` should be None, instance of Iterable or callable "
                "producing generator.")

    def __len__(self):
        # Allow some samplers have different length with `len(data_source)`,
        # such as batch sampler.
        if hasattr(self, "_length"):
            return self._length
        else:
            return len(self.data_source)

    @property
    def length(self):
        """
        Returns:
            the length of the SamplerHelper.
        """

        # since `len()` only produce integer, use length property to get None
        # for uncertain length. samplers can set length if necessary.
        try:
            length = len(self)
        except Exception:
            length = None
        return length

    @length.setter
    def length(self, length):
        self._length = length

    def apply(self, fn):
        """
        Transformations would be performed. It includes `Shuffle`, `sort`, `fit` and `shard`.
        Args:
            fn (callable): Transformations to be performed. It returns transformed iterable (and data_source).

        Returns:
            SamplerHelper: A new transformed object.
        """
        rs = fn(self)
        if isinstance(rs, (list, tuple)):
            iterable, data_source = rs
        else:
            iterable, data_source = rs, self.data_source
        sampler = type(self)(data_source, iterable)
        return sampler

    def shuffle(self, buffer_size=-1, seed=None):
        """
        Shuffle the dataset according to the given buffer size and random seed.
        Args:
            buffer_size (int): Buffer size for shuffle. if buffer_size < 0 or more than the length of the dataset, 
                buffer_size is the length of the dataset. Default: -1. 
            seed (int, optional): Seed for the random. Default: None.
        Returns:
            SamplerHelper
         """
        if seed is not None:
            random_generator = np.random.RandomState(seed)
        else:  # use the global random generator
            random_generator = np.random

        def _impl():
            buf = []
            for idx in iter(self):
                buf.append(idx)
                if buffer_size > 0 and len(buf) >= buffer_size:
                    random_generator.shuffle(buf)
                    for b in buf:
                        yield b
                    buf = []
            if len(buf) > 0:
                random_generator.shuffle(buf)
                for b in buf:
                    yield b

        return type(self)(self.data_source, _impl)

    def sort(self, cmp=None, key=None, reverse=False, buffer_size=-1):
        """
        Sort samples according to given callable cmp or key.

        Args:
            cmp (callable): The funcation of comparison. Default: None. 
            key (callable): Return element to be compared. Default: None.
            reverse (bool): If True, it means in descending order, and False means in ascending order. Default: False.
            buffer_size (int): Buffer size for sort. If buffer_size < 0 or buffer_size is more than the length of the data, 
                buffer_size will be set to the length of the data. Default: -1.
        Returns:
            SamplerHelper
        """

        def _impl():
            data_source = self.data_source
            buf = []
            for idx in iter(self):
                buf.append(idx)
                if buffer_size > 0 and len(buf) >= buffer_size:
                    buf = sorted(
                        buf,
                        cmp=(lambda x, y: cmp(x, y, data_source))
                        if cmp else None,
                        key=(lambda x: key(x, data_source)) if key else None,
                        reverse=reverse)
                    for b in buf:
                        yield b
                    buf = []
            if len(buf) > 0:
                buf = sorted(
                    buf,
                    cmp=(lambda x, y: cmp(x, y, data_source)) if cmp else None,
                    key=(lambda x: key(x, data_source)) if key else None,
                    reverse=reverse)
                for b in buf:
                    yield b

        return type(self)(self.data_source, _impl)

    def batch(self,
              batch_size,
              drop_last=False,
              batch_size_fn=None,
              batch_fn=None):
        """
        To produce a BatchSampler.

        Agrs:
            batch_size (int): Batch size.
            drop_last (bool): Whether to drop the last mini batch. Default: False.
            batch_size_fn (callable, optional): Return the size of mini batch so far. Default: None.
            batch_fn (callable, optional): Transformations to be performed. Default: None.
        Returns:
            SamplerHelper
        """
        if batch_size_fn is None:
            batch_size_fn = lambda new, count, sofar, data_source: count

        def _impl():
            data_source = self.data_source
            minibatch, size_so_far = [], 0
            for idx in iter(self):
                minibatch.append(idx)
                size_so_far = batch_size_fn(idx,
                                            len(minibatch), size_so_far,
                                            data_source)
                if size_so_far == batch_size:
                    yield minibatch
                    minibatch, size_so_far = [], 0
                elif size_so_far > batch_size:
                    yield minibatch[:-1]
                    minibatch, size_so_far = minibatch[-1:], batch_size_fn(
                        idx, 1, 0, data_source)
            if minibatch and not drop_last:
                yield minibatch

        sampler = type(self)(
            self.data_source,
            _impl) if batch_fn is None else self.apply(batch_fn)
        if batch_size_fn is None and batch_fn is None and self.length is not None:
            sampler.length = (self.length + int(not drop_last) *
                              (batch_size - 1)) // batch_size
        else:
            sampler.length = None

        return sampler

    def shard(self, num_replicas=None, rank=None):
        """
        Operates slice using multi GPU.

        Args:
            num_replicas (int, optional): The number of training process, and is also the number of GPU cards used in training. 
                Default: None.
            rank (int, optional): Number of training process. Equal to the value of the environment variable PADDLE_TRAINER_ID.
                Default: None.
        Returns:
            SamplerHelper
        """
        if num_replicas is None:
            num_replicas = ParallelEnv().nranks
        if rank is None:
            rank = ParallelEnv().local_rank

        def _impl(self):
            for i, idx in enumerate(self):
                if i % num_replicas == rank:
                    yield idx
            if i % num_replicas != num_replicas - 1 and rank > i % num_replicas:
                # use last samples to make it evenly divisible
                yield idx

        sampler = type(self)(self.data_source, _impl)
        if self.length is not None:
            sampler.length = int(math.ceil(self.length * 1.0 / num_replicas))
        else:
            sampler.length = None
        return sampler

    def list(self):
        """
        Produce a sampler with a `listiterator` when calling `iter`. Since `list`
        would fetch all contents at time, thus it can get accurate length.

        Returns:
            SamplerHelper
        """

        def _impl(sampler):
            indices = list(iter(self))
            sampler.length = len(indices)
            return iter(indices)

        return type(self)(self.data_source, _impl)


class Vocab(object):
    def __init__(self,
                 counter=None,
                 max_size=None,
                 min_freq=1,
                 token_to_idx=None,
                 unk_token='<unk>',
                 pad_token='<pad>',
                 bos_token='<bos>',
                 eos_token='<eos>',
                 **kwargs):
        # Handle special tokens
        combs = (('unk_token', unk_token), ('pad_token', pad_token),
                 ('bos_token', bos_token), ('eos_token', eos_token))
        for name, value in combs:
            kwargs[name] = value
        special_tokens = []
        special_iter = kwargs.keys()
        # sort alphabetically
        special_iter = sorted(special_iter)
        for special_token_name in special_iter:
            # Test if kwarg specifies a special token
            if not special_token_name.endswith('_token'):
                raise ValueError('{} is invalid. Only keyword arguments '
                                 'that end in \'_token\' are supported '
                                 'to declare special tokens.'.format(
                                     special_token_name))

            special_token = kwargs[special_token_name]
            if special_token is not None and special_token not in special_tokens:
                special_tokens.append(special_token)

        if counter is None:
            # use token_to_idx as dict to import pretrained vocabulary
            assert token_to_idx, (
                'token_to_idx should not be None when counter is None')
            for special_token in special_tokens:
                assert special_token in special_tokens, '{} is not in token_to_idx'.format(
                    special_token)
            self._token_to_idx = token_to_idx
            self._idx_to_token = sorted(
                self._token_to_idx.keys(),
                key=lambda token: self._token_to_idx[token])
            if unk_token:
                unk_index = self._token_to_idx[unk_token]
                self._token_to_idx = collections.defaultdict(lambda: unk_index)
                self._token_to_idx.update(token_to_idx)
        else:
            self._idx_to_token = list(special_tokens)
            self._token_to_idx = collections.defaultdict()
            self._token_to_idx.update(
                (token, idx) for idx, token in enumerate(self._idx_to_token))
            self._index_counter_keys(counter, special_tokens, max_size,
                                     min_freq)
            if token_to_idx:
                self._sort_index_according_to_user_specification(token_to_idx)
            if unk_token:
                self._token_to_idx.default_factory = lambda: self._token_to_idx[unk_token]

        # _expose_tokens_as_attributes
        self._identifiers_to_tokens = kwargs
        for identifier, token in kwargs.items():
            if identifier.startswith('_'):
                raise ValueError(
                    'It is not allowed to use identifiers starting with '
                    'underscore. In Python identifier names beginning with '
                    'underscore are internal.')
            if hasattr(self, identifier):
                raise ValueError(
                    'vocab.{} already exists. '
                    'Please choose a different identifier for token {}'.format(
                        identifier, token))
            setattr(self, identifier, token)

    def _index_counter_keys(self, counter, special_tokens, max_size, min_freq):
        # sort by frequency, then alphabetically
        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        special_tokens = set(special_tokens)
        max_size = None if max_size is None else max_size + len(special_tokens)
        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == max_size:
                break
            if token not in special_tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    def _sort_index_according_to_user_specification(self, token_to_idx):
        # Sanity checks
        if not set(token_to_idx.keys()).issubset(self.token_to_idx.keys()):
            raise ValueError(
                'User-specified token_to_idx mapping can only contain '
                'tokens that will be part of the vocabulary.')
        if len(set(token_to_idx.values())) != len(token_to_idx):
            raise ValueError(
                'User-specified indices must not contain duplicates.')
        if min(token_to_idx.values()) < 0 or max(token_to_idx.values()) >= len(
                self.token_to_idx):
            raise ValueError(
                'User-specified indices must not be < 0 or >= the number of tokens '
                'that will be in the vocabulary. The current vocab contains {}'
                'tokens.'.format(len(self.token_to_idx)))

        # Update index ordering
        for token, new_idx in token_to_idx.items():
            old_idx = self.token_to_idx[token]
            ousted_token = self.idx_to_token[new_idx]

            self.token_to_idx[token] = new_idx
            self.token_to_idx[ousted_token] = old_idx
            self.idx_to_token[old_idx] = ousted_token
            self.idx_to_token[new_idx] = token

    def to_tokens(self, indices):
        to_reduce = False
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
            to_reduce = True

        max_idx = len(self._idx_to_token) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx:
                raise ValueError(
                    'Token index {} in the provided `indices` is invalid.'.
                    format(idx))
            tokens.append(self._idx_to_token[idx])

        return tokens[0] if to_reduce else tokens

    def to_indices(self, tokens):
        return self[tokens]

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[tokens]
        else:
            return [self._token_to_idx[token] for token in tokens]

    def __len__(self):
        return len(self._idx_to_token)

    def __contains__(self, token):
        return token in self._token_to_idx

    def __call__(self, tokens):
        return self[tokens]

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def token_to_idx(self):
        return self._token_to_idx

    def to_json(self, path=None):
        vocab_dict = {}
        vocab_dict['idx_to_token'] = self.idx_to_token
        vocab_dict['token_to_idx'] = dict(self.token_to_idx)
        vocab_dict['unk_token'] = self.unk_token
        vocab_dict['identifiers_to_tokens'] = self._identifiers_to_tokens
        json_str = json.dumps(vocab_dict)
        if path:
            with io.open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str):
        if os.path.isfile(json_str):
            with io.open(json_str, 'w', encoding='utf-8') as f:
                vocab_dict = json.load(f)
        else:
            vocab_dict = json.loads(json_str)
        token_to_idx = vocab_dict.get('token_to_idx')
        unk_token = vocab_dict.get('unk_token')
        identifiers_to_tokens = vocab_dict.get('identifiers_to_tokens', dict())

        vocab = cls(counter=None,
                    token_to_idx=token_to_idx,
                    unk_token=unk_token,
                    **identifiers_to_tokens)
        return vocab

    @classmethod
    def from_dict(cls, token_to_idx, unk_token=None, **kwargs):
        vocab = cls(counter=None,
                    token_to_idx=token_to_idx,
                    unk_token=unk_token,
                    **kwargs)
        return vocab

    @staticmethod
    def build_vocab(iterator,
                    max_size=None,
                    min_freq=1,
                    token_to_idx=None,
                    unk_token='<unk>',
                    pad_token='<pad>',
                    bos_token='<bos>',
                    eos_token='<eos>',
                    **kwargs):
        counter = collections.Counter()
        for tokens in iterator:
            counter.update(tokens)
        vocab = Vocab(
            counter,
            max_size=max_size,
            min_freq=min_freq,
            token_to_idx=token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs)
        return vocab


@six.add_metaclass(InitTrackerMeta)
class PreTrainedTokenizer(object):
    """
    The base class of the BertTokenizer, which provides the interface for 
    loading and saving the tokenzier used in the pre-raining.
    """
    tokenizer_config_file = "tokenizer_config.json"
    pretrained_init_configuration = {}
    resource_files_names = {}  # keys are arguments of __init__
    pretrained_resource_files_map = {}

    def _wrap_init(self, original_init, *args, **kwargs):
        # expose tokens as attributes
        if hasattr(inspect, 'getfullargspec'):
            (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _,
             _) = inspect.getfullargspec(original_init)
        else:
            (spec_args, spec_varargs, spec_varkw,
             spec_defaults) = inspect.getargspec(original_init)
        init_dict = dict(zip(spec_args, args))
        kwargs_dict = dict(
            zip(spec_args[-len(spec_defaults):],
                spec_defaults)) if spec_defaults else {}
        kwargs_dict.update(kwargs)
        init_dict.update(kwargs_dict)
        for identifier, token in init_dict.items():
            if identifier.endswith('_token'):
                setattr(self, identifier, token)

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a sequence of tokens into ids using the vocab.
        Args：
            tokens (list(str)): List of tokens.
        Returns:
            list: Converted id list.

        """
        return [self.vocab[token] for token in tokens]

    def convert_tokens_to_string(self, tokens):
        """ 
        Converts a sequence of tokens (string) in a single string.
        The most simple way to do it is ' '.join(self.convert_ids_to_tokens(token_ids))
        but we often want to remove sub-word tokenization artifacts at the same time.
        Args:
            tokens (list(str)): List of tokens.
        Returns:
            str: Converted string.
        """
        return " ".join(tokens)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs,
                        **kwargs):
        """
        Load tokenizer from pretrained model.
        Args:
            pretrained_model_name_or_path (str): A name or a path of pre-trained model.
            *init_inputs (tuple): The additional init inputs.
            **kwargs (dict): The Additional inputs.
        Returns:
            PreTrainedTokenizer
        """
        pretrained_models = list(cls.pretrained_init_configuration.keys())
        vocab_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in pretrained_models:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(
                cls.pretrained_init_configuration[
                    pretrained_model_name_or_path])
        else:
            if os.path.isdir(pretrained_model_name_or_path):
                for file_id, file_name in cls.resource_files_names.items():
                    full_file_name = os.path.join(pretrained_model_name_or_path,
                                                  file_name)
                    vocab_files[file_id] = full_file_name
                vocab_files["tokenizer_config_file"] = os.path.join(
                    pretrained_model_name_or_path, cls.tokenizer_config_file)
            else:
                raise ValueError(
                    "Calling {}.from_pretrained() with a model identifier or the "
                    "path to a directory instead. The supported model "
                    "identifiers are as follows: {}".format(
                        cls.__name__, cls.pretrained_init_configuration.keys()))

        default_root = os.path.join(DATA_HOME, pretrained_model_name_or_path)
        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            resolved_vocab_files[
                file_id] = file_path if file_path is None or os.path.isfile(
                    file_path) else get_path_from_url(file_path, default_root,
                                                      None)

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop(
            "tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with io.open(tokenizer_config_file, encoding="utf-8") as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration

        # position args are stored in kwargs, maybe better not include
        saved_init_inputs = init_kwargs.pop("init_inputs", ())
        if not init_inputs:
            init_inputs = saved_init_inputs

        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        # Merge resolved_vocab_files arguments in init_kwargs if not including.
        # Maybe need more ways to load resources.
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path

        tokenizer = cls(*init_inputs, **init_kwargs)
        return tokenizer

    def save_pretrained(self, save_directory):
        """
        Save tokenizer config and resources to files.
        Args:
            save_directory (str): Directory to store token configuration file and vocab.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving directory ({}) should be a directory".format(save_directory)
        tokenizer_config_file = os.path.join(save_directory,
                                             self.tokenizer_config_file)
        tokenizer_config = self.init_config
        with io.open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

        self.save_resources(save_directory)

    def save_resources(self, save_directory):
        """
        Save resources to a file.
        Args:
            save_directory (str): Directory to store resources.
        """
        assert hasattr(self, 'vocab') and len(
            self.resource_files_names) == 1, "Must overwrite `save_resources`"
        file_name = os.path.join(save_directory,
                                 list(self.resource_files_names.values())[0])
        self.save_vocabulary(file_name, self.vocab)

    @staticmethod
    def load_vocabulary(filename, unk_token=None, **kwargs):
        """
        Loads a vocabulary file into a dictionary.
        Args:
            filename (str): File path to load.
            unk_token (str, optional): UNK token. Default: None.
            **kwargs (dict): The additional inputs for Vocab.from_dict.
        Returns:
            Vocab: vocab.
        """
        token_to_idx = {}
        with io.open(filename, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                token = line.rstrip('\n')
                token_to_idx[token] = int(index)
        vocab = Vocab.from_dict(token_to_idx, unk_token=unk_token, **kwargs)
        return vocab

    @staticmethod
    def save_vocabulary(filename, vocab):
        """
        Save all tokens to a vocabulary file.
        Agrs:
            filename (str): File path to be saved.
            vocab (Vocab|dict): Vocab to be saved.
        """
        if isinstance(vocab, Vocab):
            tokens = vocab.idx_to_token
        else:
            tokens = sorted(vocab.keys(), key=lambda token: vocab[token])
        with io.open(filename, 'w', encoding='utf-8') as f:
            for token in tokens:
                f.write(token + '\n')


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming utf-8 input.
    Args:
        text (str|bytes): Text to be converted to unicode.
    Returns: 
        str: converted text.
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def whitespace_tokenize(text):
    """
    Runs basic whitespace cleaning and splitting on a peice of text.
    Args:
        text (str): Text to be tokened.
    Returns:
        list(str): Token list.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(char):
    """
    Checks whether `chars` is a whitespace character.
    """
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class BertBasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (bool): Whether to convert the input to lowercase. Default: True.
    """

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer."""

        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """
        Tokenizes a piece of text using basic tokenizer.

        Args:
            text (str): A piece of text.

        Returns: 
            list(str): A list of tokens.
        """
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """
        Strips accents from a piece of text.
        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """
        Splits punctuation on a piece of text.
        """
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """
        Adds whitespace around any CJK character.
        """
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """
        Checks whether CP is the codepoint of a CJK character.
        """

        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """
        Performs invalid character removal and whitespace cleanup on text.
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """
    Runs WordPiece tokenization.
    Args:
        vocab (Vocab): Vocab of the word piece tokenizer.
        unk_token (str):  A specific token to replace all unkown tokens.
        max_input_chars_per_word (int):  If a word's length is more than max_input_chars_per_word, it will be 
            dealt as unknown word. Default: 100.
    """

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer`.

        Returns:
            list (str): A list of wordpiece tokens.

        Example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BertTokenizer(PreTrainedTokenizer):
    """
    Runs bert tokenization, including BertBasicTokenize and WordpieceTokenizer.
    Args:
        vocab_file (str): filename of the vocab
        do_lower_case (bool): Whether to convert the input to lowercase. Default: True.
        unk_token (str): A specific token for unkown words. Default: "[UNK]".
        sep_token (str): A specific token for separator token . Default: "[SEP]".
        pad_token (str): A specific token for padding. Default: "[PAD]".
        cls_token (str): A specific token for cls. Default: "[CLS]".
        mask_token (str): A specific token for mask. Default: "[MASK]".

    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "bert-base-uncased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-uncased-vocab.txt",
            "bert-large-uncased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-large-uncased-vocab.txt",
            "bert-base-cased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-cased-vocab.txt",
            "bert-large-cased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-large-cased-vocab.txt",
            "bert-base-multilingual-uncased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-multilingual-uncased-vocab.txt",
            "bert-base-multilingual-cased":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-multilingual-cased-vocab.txt",
            "bert-base-chinese":
            "https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-chinese-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "bert-base-uncased": {
            "do_lower_case": True
        },
        "bert-large-uncased": {
            "do_lower_case": True
        },
        "bert-base-cased": {
            "do_lower_case": False
        },
        "bert-large-cased": {
            "do_lower_case": False
        },
        "bert-base-multilingual-uncased": {
            "do_lower_case": True
        },
        "bert-base-multilingual-cased": {
            "do_lower_case": False
        },
        "bert-base-chinese": {
            "do_lower_case": False
        }
    }

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.basic_tokenizer = BertBasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=unk_token)

    @property
    def vocab_size(self):
        """
        return size of the vocab.
        """
        return len(self.vocab)

    def _tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def __call__(self, text):
        """
        Return list of tokens of text.
        """
        return self._tokenize(text)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string.
        Args:
            tokens (list): Tokens to be converted.
        Returns:
            str: Converted string from tokens.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string  # 
