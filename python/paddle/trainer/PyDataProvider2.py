# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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

import cPickle
import logging


class SequenceType(object):
    NO_SEQUENCE = 0
    SEQUENCE = 1
    SUB_SEQUENCE = 2


# TODO(yuyang18): Add string data type here.
class DataType(object):
    Dense = 0
    SparseNonValue = 1
    SparseValue = 2
    Index = 3


class CacheType(object):
    NO_CACHE = 0  # No cache at all

    # First pass, read data from python.  And store them in memory. Read from
    # memory during rest passes.
    CACHE_PASS_IN_MEM = 1


class InputType(object):
    __slots__ = ['dim', 'seq_type', 'type']

    def __init__(self, dim, seq_type, tp):
        self.dim = dim
        self.seq_type = seq_type
        self.type = tp


def dense_slot(dim, seq_type=SequenceType.NO_SEQUENCE):
    return InputType(dim, seq_type, DataType.Dense)


def sparse_non_value_slot(dim, seq_type=SequenceType.NO_SEQUENCE):
    return InputType(dim, seq_type, DataType.SparseNonValue)


def sparse_value_slot(dim, seq_type=SequenceType.NO_SEQUENCE):
    return InputType(dim, seq_type, DataType.SparseValue)


def index_slot(dim, seq_type=SequenceType.NO_SEQUENCE):
    return InputType(dim, seq_type, DataType.Index)


dense_vector = dense_slot
sparse_binary_vector = sparse_non_value_slot
sparse_vector = sparse_value_slot
integer_value = index_slot

def dense_vector_sequence(dim):
    return dense_vector(dim, seq_type=SequenceType.SEQUENCE)

def dense_vector_sub_sequence(dim):
    return dense_vector(dim, seq_type=SequenceType.SUB_SEQUENCE)

def sparse_binary_vector_sequence(dim):
    return sparse_binary_vector(dim, seq_type=SequenceType.SEQUENCE)

def sparse_binary_vector_sub_sequence(dim):
    return sparse_binary_vector(dim, seq_type=SequenceType.SUB_SEQUENCE)

def sparse_vector_sequence(dim):
    return sparse_vector(dim, seq_type=SequenceType.SEQUENCE)

def sparse_vector_sub_sequence(dim):
    return sparse_vector(dim, seq_type=SequenceType.SUB_SEQUENCE)

def integer_value_sequence(dim):
    return integer_value(dim, seq_type=SequenceType.SEQUENCE)

def integer_value_sub_sequence(dim):
    return integer_value(dim, seq_type=SequenceType.SUB_SEQUENCE)

def integer_sequence(dim):
    return index_slot(dim, seq_type=SequenceType.SEQUENCE)


class SingleSlotWrapper(object):
    def __init__(self, generator):
        self.generator = generator

    def __call__(self, obj, filename):
        for item in self.generator(obj, filename):
            yield [item]


def provider(input_types=None, should_shuffle=True, pool_size=-1,
             can_over_batch_size=True,
             calc_batch_size=None,
             cache=CacheType.NO_CACHE,
             init_hook=None, **kwargs):
    """
    Provider decorator. Use it to make a function into PyDataProvider2 object.
    In this function, user only need to get each sample for some train/test
    file.

    The basic usage is:

    ..  code-block:: python

        @provider(some data provider config here...)
        def process(settings, file_name):
            while not at end of file_name:
                sample = readOneSampleFromFile(file_name)
                yield sample.

    The configuration of data provider should be setup by\:

    :param input_types: Specify the input types, can also be set in init_hook.
                        It is a list of InputType object. For example, input_types= \
                        [dense_vector(9), integer_value(2)].
    :param should_shuffle: True if data should shuffle.
    :type should_shuffle: bool
    :param pool_size: Max number of sample in data pool.
    :type pool_size: int
    :param can_over_batch_size: True if paddle can return a mini-batch larger
                                than batch size in settings. It is useful when
                                custom calculate one sample's batch_size.

                                It is very danger to set it to false and use
                                calc_batch_size together. Default is false.
    :param calc_batch_size: a method to calculate each sample's batch size.
                            Default each sample's batch size is 1. But to you
                            can customize each sample's batch size.
    :param cache: Cache strategy of Data Provider. Default is CacheType.NO_CACHE

    :param init_hook: Initialize hook. Useful when data provider need load some
                      external data like dictionary. The parameter is
                      (settings, file_list, \*\*kwargs).

                      - settings\: Is the global settings. User can set
                                   settings.input_types here.
                      - file_list\: All file names for passed to data provider.
                      - kwargs: Other keyword arguments passed from
                        trainer_config's args parameter.
    """

    def __wrapper__(generator):
        class DataProvider(object):
            def __init__(self, file_list, **kwargs):
                self.logger = logging.getLogger("")
                self.logger.setLevel(logging.INFO)
                self.input_types = None
                if 'slots' in kwargs:
                    self.logger.warning('setting slots value is deprecated, '
                                        'please use input_types instead.')
                    self.slots = kwargs['slots']
                self.slots = input_types
                self.should_shuffle = should_shuffle
                self.pool_size = pool_size
                self.can_over_batch_size = can_over_batch_size
                self.calc_batch_size = calc_batch_size
                self.file_list = file_list
                self.generator = generator
                self.cache = cache
                if init_hook is not None:
                    init_hook(self, file_list=file_list, **kwargs)
                if self.input_types is not None:
                    self.slots = self.input_types
                assert self.slots is not None
                assert self.generator is not None

                if len(self.slots) == 1:
                    self.generator = SingleSlotWrapper(self.generator)

        return DataProvider

    return __wrapper__


def deserialize_args(args):
    """
    Internal use only.
    :param args:
    :return:
    """
    return cPickle.loads(args)
