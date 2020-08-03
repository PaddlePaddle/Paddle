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

from .utils import InitTrackerMeta

# __all__ = ['Stack', 'Pad', 'Tuple']  # batchify
# __all__ += ['']  # dataset helper, sampler helper
# __all__ += ['']  # transform


class Stack(object):
    """
    Stack the input data samples to construct the batch.
    The N input samples must have the same shape/length and will be stacked to construct a batch.

    Args:
        dtype (str|numpy.dtype): The value type of the output. If it is set to None, the input data type is used. Default: None.

    Example:
        .. code-block:: python:

            from paddle.incubate.hapi.text import Stack
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
            data (list) – The input data samples
        Returns:
            NDArray: Batch_data

        """
        data = np.stack(data).astype(self._dtype) if self._dtype else np.stack(
            data)
        return data


class Pad(object):
    """
    Return a callable that pads and stacks data.

    Args:
        axis (int): The axis to pad the arrays. The arrays will be padded to the largest dimension at axis. For example, 
            assume the input arrays have shape (10, 8, 5), (6, 8, 5), (3, 8, 5) and the axis is 0. 
            Each input will be padded into (10, 8, 5) and then stacked to form the final output, which has shape（3, 10, 8, 5). Default: 0.
        pad_val (float|int): The padding value. Default: 0.
        ret_length (bool): Whether to return the valid length in the output. Default: False.
        dtype (str|numpy.dtype): The value type of the output. If it is set to None, the input data type is used. Default: None.
        round_to (int): If specified, the padded dimension will be rounded to be multiple of this argument. Default: None.

        Example:
            .. code-block:: python:
                from paddle.incubate.hapi.text import Pad
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
        The input can be list of numpy.ndarray, list of numbers or list of mxnet.nd.NDArray. 
        Inputting mxnet.nd.NDArray is discouraged as each array need to be converted to numpy for efficient padding.

        The arrays will be padded to the largest dimension at axis and then stacked to form the final output. In addition, the function will output the original dimensions at the axis if ret_length is turned on.

        Args:
            data (List[np.ndarray]|List[List[dtype]]|List[mx.nd.NDArray]): List of samples to pad and stack.

        Returns:
            NDArray: Batch_data, data in the minibatch. Shape is (N, …)
            NDArray (optional): Valid_length, the sequences’ original lengths at the padded axis. Shape is (N,). 
                This will only be returned in ret_length is True.
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
    Each data sample should be a list or tuple containing multiple attributes. The i`th batchify function stored in `Tuple will be applied on the i`th attribute. For example, each data sample is (nd_data, label). You can wrap two batchify functions using `Tuple(DataBatchify, LabelBatchify) to batchify nd_data and label correspondingly.

    Args:
        fn (list|tuple|callable): The batchify functions to wrap.
        *args (tuple of callable): The additional batchify functions to wrap.
    Example:
        .. code-block:: python:
            from paddle.incubate.hapi.text import Tuple, Pad, Stack
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
            assert isinstance(ele_fn, callable), 'Batchify functions must be callable! ' \
                                                'type(fn[%d]) = %s' % (i, str(type(ele_fn)))

    def __call__(self, data):
        """
        Batchify the input data.

        Args:
            list: The samples to batchfy. Each sample should contain N attributes.
        Returns:
            tuple: A tuple of length N. Contains the batchified result of each attribute in the input.
        
        """
        assert len(data[0]) == len(self._fn),\
            'The number of attributes in each data sample should contains' \
            ' {} elements'.format(len(self._fn))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
        return tuple(ret)


class WrapDataset(Dataset):
    """
    decorates dataset with shuffle, sort and other transformations.
    It acts as some specific sampler or iterator for dataset, and
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._transform_func = None

    def __iter__(self):
        for idx in range(len(self.dataset)):
            yield self.dataset[idx]

    def __getitem__(self, idx):
        return self._transform_func(self.dataset[
            idx]) if self._transform_func else self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def shuffle(self, buffer_size=-1, seed=None):
        """Shuffle the dataset."""
        if seed is not None:
            np_rand_state_bak = np.random.get_state()
            np.random.seed(seed)
        buffer_size = len(self.dataset) if buffer_size < 0 else buffer_size
        for i in range(0, len(self.dataset), buffer_size):
            buf = list(range(i, i + buffer_size))
            np.random.shuffle(buf)
            self.dataset[i:i + buffer_size] = [self.dataset[idx] for idx in buf]
        if seed is not None:
            np.random.set_state(np_rand_state_bak)
        return self

    def sort(self, cmp=None, key=None, reverse=False, buffer_size=-1):
        """Sort the dataset. """
        buffer_size = len(self.dataset) if buffer_size < 0 else buffer_size
        for i in range(0, len(self.dataset), buffer_size):
            self.dataset[i:i + buffer_size] = sorted(
                range(i, i + buffer_size),
                cmp=lambda x, y: cmp(self.dataset[x], self.dataset[y]) if cmp else None,
                key=lambda x: key(self.dataset[x]) if key else None,
                reverse=reverse)
        return self

    def filter(self, predicate_func):
        """Filter the dataset by predicate_func."""
        self.dataset = [
            self.dataset[idx] for idx in range(len(self.dataset))
            if predicate_func(self.dataset[idx])
        ]
        return self

    def apply(self, transform_func, lazy=False):
        """
        If `lazy`, transformations would be delayed and performed when calling
        `__getitem__`, which would run multi-times for each sample but can take
        advantage of DataLoader multi-processing. `transform_func` receives single
        sample as argument rather than dataset.
        """
        if lazy:
            self._transform_func = transform_func
        else:
            self.dataset = [
                transform_func(self.dataset[idx])
                for idx in range(len(self.dataset))
            ]
        return self

    def shard(self, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = ParallelEnv().nranks
        if rank is None:
            rank = ParallelEnv().local_rank
        num_samples = int(math.ceil(len(self.dataset) * 1.0 / num_replicas))
        total_size = num_samples * num_replicas
        # add extra samples to make it evenly divisible
        self.dataset = [
            self.dataset[idx]
            for idx in range(len(self.dataset)) + range(total_size - len(
                self.dataset)) if idx % num_replicas == rank
        ]
        return self


class SamplerHelper(object):
    """
    Sampler Factory. chain of sampling strategies

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    Also can be used as batch iterator instead of indices iterator when `iterator`
    yield samples rather than indices by initializing `iterator` with a iterable
    dataset.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`DataLoader`, but is expected in any
              calculation involving the length of a :class:`DataLoader`.
    Args:
        dataset
        iterable

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
        """Transformations would be performed."""
        rs = fn(self)
        if isinstance(rs, (list, tuple)):
            iterable, data_source = rs
        else:
            iterable, data_source = rs, self.data_source
        sampler = type(self)(data_source, iterable)
        return sampler

    def shuffle(self, buffer_size=-1, seed=None):
        """Shuffle the samples according to the seed."""
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
        """Sort samples"""
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
            batch_size (int):
            drop_last
            batch_size_fn
            batch_fn
        Returns:
            BatchSampler
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

        Args：
            num_replicas
            rank
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
        """

        def _impl(sampler):
            indices = list(iter(self))
            sampler.length = len(indices)
            return iter(indices)

        return type(self)(self.data_source, _impl)


@six.add_metaclass(InitTrackerMeta)
class PreTrainedTokenizer(object):
    """
    The base class of the pre-training tokenizer, which provides the interface for 
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
            tokens (list(str)): list of tokens
        Returns:
            list: converted id list.

        """
        return [self.vocab[token] for token in tokens]

    def convert_tokens_to_string(self, tokens):
        """ 
        Converts a sequence of tokens (string) in a single string.
        The most simple way to do it is ' '.join(self.convert_ids_to_tokens(token_ids))
        but we often want to remove sub-word tokenization artifacts at the same time.
        Args:
            tokens (list(str)): list of tokens
        Returns:
            str: converted string.
        """
        return " ".join(tokens)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs,
                        **kwargs):
        """
        get tokenizer from pretrained model.
        Args:
            cls:
            pretrained_model_name_or_path (str):
            *init_inputs:
            kwargs:
        Returns:
            tokenizer
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
            save_directory (str): directory to be saved
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
            save_directory (str):
        """
        assert hasattr(self, 'vocab') and len(
            self.resource_files_names) == 1, "Must overwrite `save_resources`"
        file_name = os.path.join(save_directory,
                                 list(self.resource_files_names.values())[0])
        self.save_vocabulary(file_name, self.vocab)

    @staticmethod
    def load_vocabulary(filename, unk_token=None, **kwargs):
        """
        Load the vocabulary from a file.
        Args:
            filename (str): file path to load.
            unk_token (str|None): unk token. Default: None.
            kwargs (dict): 

        Returns:
            Vocab
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
            filename (str): file path to be saved.
            vocab (Vocab): Vocab to be saved
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
        text (str|bytes): text to be converted to utf-8.
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
        text (str): text to be processed.
    Returns:
        list(str): processed token list.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
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
        do_lower_case (bool):Whether to convert the input to lowercase. Default: True.
    """

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer."""

     
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text.
        Args:
            text (str):

        Returns:


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
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
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
        """Adds whitespace around any CJK character."""
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
        """Checks whether CP is the codepoint of a CJK character."""
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
        """Performs invalid character removal and whitespace cleanup on text."""
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
        vocab
        unk_token (str):
        max_input_chars_per_word (int):  Default: 100.
    """

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer`.

        Example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        Returns:
            A list of wordpiece tokens.
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
    Runs bert tokenization.
    Args:
        vocab_file (str): filename of the vocab
        do_lower_case (bool): Whether to convert the input to lowercase. Default: True.
        unk_token (str): Default: "[UNK]".
        sep_token (str): Default: "[SEP]".
        pad_token (str): Default: "[PAD]".
        cls_token (str): Default: "[CLS]".
        mask_token (str): Default: "[MASK]".
        kwargs (dict): 

    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "bert-base-uncased":
            "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
            "bert-large-uncased":
            "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
            "bert-base-cased":
            "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
            "bert-large-cased":
            "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
            "bert-base-multilingual-uncased":
            "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
            "bert-base-multilingual-cased":
            "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
            "bert-base-chinese":
            "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
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
        """return size of the vocab."""
        return len(self.vocab)

    def _tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def __call__(self, text):
        """return list of tokens of text."""
        return self._tokenize(text)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string.
        Args:
            tokens (list): tokens to be converted.
        Returns:
            str: converted string from tokens.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string
