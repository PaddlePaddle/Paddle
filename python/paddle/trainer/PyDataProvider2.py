# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
import collections
import functools
import itertools

logging.basicConfig(format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s]"
                    " %(message)s")


class SequenceType(object):
    NO_SEQUENCE = 0
    SEQUENCE = 1
    SUB_SEQUENCE = 2

    @classmethod
    def tostring(cls, value):
        for k in cls.__dict__:
            if not k.startswith('__'):
                if getattr(cls, k) == value:
                    return cls.__name__ + '.' + k
        return 'INVALID(' + str(value) + ')'


# TODO(yuyang18): Add string data type here.
class DataType(object):
    Dense = 0
    SparseNonValue = 1
    SparseValue = 2
    Index = 3

    @classmethod
    def tostring(cls, value):
        for k in cls.__dict__:
            if not k.startswith('__'):
                if getattr(cls, k) == value:
                    return cls.__name__ + '.' + k
        return 'INVALID(' + str(value) + ')'


class CacheType(object):
    NO_CACHE = 0  # No cache at all

    # First pass, read data from python.  And store them in memory. Read from
    # memory during rest passes.
    CACHE_PASS_IN_MEM = 1


class InputType(object):
    """
    InputType is the base class for paddle input types.

    ..  note::

        this is a base class, and should never be used by user.

    :param dim: dimension of input. If the input is an integer, it means the
                value range. Otherwise, it means the size of layer.
    :type dim: int
    :param seq_type: sequence type of input. 0 means it is not a sequence. 1
                     means it is a variable length sequence. 2 means it is a
                     nested sequence.
    :type seq_type: int
    :param type: data type of input.
    :type type: int
    """
    __slots__ = ['dim', 'seq_type', 'type']

    def __init__(self, dim, seq_type, tp):
        self.dim = dim
        self.seq_type = seq_type
        self.type = tp

    def __repr__(self):
        """
        Return a human readable representation like 'InputType(dim=25921, 
            seq_type=SequenceType.NO_SEQUENCE, type=DataType.Dense)'
        """
        repr_str = type(self).__name__
        repr_str += '('
        serialize_func_map = {
            'dim': repr,
            'seq_type': SequenceType.tostring,
            'type': DataType.tostring
        }
        for idx, k in enumerate(self.__slots__):
            if idx != 0:
                repr_str += ', '
            repr_str += (
                k + '=' + serialize_func_map.get(k, repr)(getattr(self, k)))
        repr_str += ')'
        return repr_str


def dense_slot(dim, seq_type=SequenceType.NO_SEQUENCE):
    """
    Dense Array. It means the input feature is dense array with float type.
    For example, if the input is an image with 28*28 pixels, the input of
    Paddle neural network could be a dense vector with dimension 784 or a
    numpy array with shape (28, 28).

    For the 2-D convolution operation, each sample in one mini-batch must have
    the similarly size in PaddlePaddle now. But, it supports variable-dimension
    feature across mini-batch. For the variable-dimension, the param dim is not
    used. While the data reader must yield numpy array and the data feeder will
    set the data shape correctly.

    :param dim: dimension of this vector.
    :type dim: int
    :param seq_type: sequence type of input.
    :type seq_type: int
    :return: An input type object.
    :rtype: InputType
    """
    return InputType(dim, seq_type, DataType.Dense)


def sparse_non_value_slot(dim, seq_type=SequenceType.NO_SEQUENCE):
    """
    Sparse binary vector. It means the input feature is a sparse vector and the
    every element in this vector is either zero or one.

    :param dim: dimension of this vector.
    :type dim: int
    :param seq_type: sequence type of this input.
    :type seq_type: int
    :return: An input type object.
    :rtype: InputType
    """
    return InputType(dim, seq_type, DataType.SparseNonValue)


def sparse_value_slot(dim, seq_type=SequenceType.NO_SEQUENCE):
    """
    Sparse vector. It means the input feature is a sparse vector. Most of the
    elements in this vector are zero, others could be any float value.

    :param dim: dimension of this vector.
    :type dim: int
    :param seq_type: sequence type of this input.
    :type seq_type: int
    :return: An input type object.
    :rtype: InputType
    """
    return InputType(dim, seq_type, DataType.SparseValue)


def index_slot(value_range, seq_type=SequenceType.NO_SEQUENCE):
    """
    Data type of integer.

    :param seq_type: sequence type of this input.
    :type seq_type: int
    :param value_range: range of this integer.
    :type value_range: int
    :return: An input type object
    :rtype: InputType
    """
    return InputType(value_range, seq_type, DataType.Index)


dense_vector = dense_slot
sparse_binary_vector = sparse_non_value_slot
sparse_float_vector = sparse_value_slot
integer_value = index_slot

# dense_array can be used for variable-length input feature.
# Each feature is not a vector, but a multi-dimensional array.
dense_array = dense_slot


def dense_vector_sequence(dim):
    """
    Data type of a sequence of dense vector.

    :param dim: dimension of dense vector.
    :type dim: int
    :return: An input type object
    :rtype: InputType
    """
    return dense_vector(dim, seq_type=SequenceType.SEQUENCE)


def dense_vector_sub_sequence(dim):
    return dense_vector(dim, seq_type=SequenceType.SUB_SEQUENCE)


def sparse_binary_vector_sequence(dim):
    """
    Data type of a sequence of sparse vector, which every element is either zero
     or one.

    :param dim: dimension of sparse vector.
    :type dim: int
    :return: An input type object
    :rtype: InputType
    """
    return sparse_binary_vector(dim, seq_type=SequenceType.SEQUENCE)


def sparse_binary_vector_sub_sequence(dim):
    return sparse_binary_vector(dim, seq_type=SequenceType.SUB_SEQUENCE)


def sparse_float_vector_sequence(dim):
    """
    Data type of a sequence of sparse vector, which most elements are zero,
    others could be any float value.

    :param dim: dimension of sparse vector.
    :type dim: int
    :return: An input type object
    :rtype: InputType
    """
    return sparse_float_vector(dim, seq_type=SequenceType.SEQUENCE)


def sparse_float_vector_sub_sequence(dim):
    return sparse_float_vector(dim, seq_type=SequenceType.SUB_SEQUENCE)


def integer_value_sequence(value_range):
    """
    Data type of a sequence of integer.

    :param value_range: range of each element.
    :type value_range: int
    """
    return integer_value(value_range, seq_type=SequenceType.SEQUENCE)


def integer_value_sub_sequence(dim):
    return integer_value(dim, seq_type=SequenceType.SUB_SEQUENCE)


integer_sequence = integer_value_sequence


class SingleSlotWrapper(object):
    def __init__(self, generator):
        self.generator = generator

    def __call__(self, obj, filename):
        for item in self.generator(obj, filename):
            if isinstance(item, dict):
                yield item
            else:
                yield [item]


class InputOrderWrapper(object):
    def __init__(self, generator, input_order):
        self.generator = generator
        self.input_order = input_order

    def __call__(self, obj, filename):
        for item in self.generator(obj, filename):
            if isinstance(item, dict):
                yield [
                    item.get(input_name, None)
                    for input_name in self.input_order
                ]
            else:
                yield item


class CheckWrapper(object):
    def __init__(self, generator, input_types, check_fail_continue, logger):
        self.generator = generator
        self.input_types = input_types
        self.check_fail_continue = check_fail_continue
        self.logger = logger

    def __call__(self, obj, filename):
        for items in self.generator(obj, filename):
            try:
                assert len(items) == len(self.input_types)
                assert len(filter(lambda x: x is None, items)) == 0
                for item, input_type in itertools.izip(items, self.input_types):
                    callback = functools.partial(CheckWrapper.loop_callback,
                                                 input_type)

                    for _ in xrange(input_type.seq_type):
                        callback = functools.partial(CheckWrapper.loop_check,
                                                     callback)
                    callback(item)

                yield items
            except AssertionError as e:
                self.logger.warning(
                    "Item (%s) is not fit the input type with error %s" %
                    (repr(item), repr(e)))

                if self.check_fail_continue:
                    continue
                else:
                    raise

    @staticmethod
    def loop_callback(input_type, each):
        assert isinstance(input_type, InputType)
        if input_type.type == DataType.Dense:
            assert isinstance(each, collections.Sequence)
            for d in each:
                assert isinstance(d, float)
            assert len(each) == input_type.dim
        elif input_type.type == DataType.Index:
            assert isinstance(each, int)
            assert each < input_type.dim
        elif input_type.type == DataType.SparseNonValue \
                or input_type.type == DataType.SparseValue:
            assert isinstance(each, collections.Sequence)
            sparse_id = set()
            for k in each:
                if input_type.type == DataType.SparseValue:
                    k, v = k
                    assert isinstance(v, float)
                assert isinstance(k, int)
                assert k < input_type.dim
                sparse_id.add(k)
            assert len(sparse_id) == len(each)
        else:
            raise RuntimeError("Not support input type")

    @staticmethod
    def loop_check(callback, item):
        for each in item:
            callback(each)


class CheckInputTypeWrapper(object):
    def __init__(self, generator, input_types, logger):
        self.generator = generator
        self.input_types = input_types
        self.logger = logger

    def __call__(self, obj, filename):
        for items in self.generator(obj, filename):
            try:
                # dict type is required for input_types when item is dict type
                assert (isinstance(items, dict) and \
                        not isinstance(self.input_types, dict))==False
                yield items
            except AssertionError as e:
                self.logger.error(
                    "%s type is required for input type but got %s" %
                    (repr(type(items)), repr(type(self.input_types))))
                raise


def provider(input_types=None,
             should_shuffle=None,
             pool_size=-1,
             min_pool_size=-1,
             can_over_batch_size=True,
             calc_batch_size=None,
             cache=CacheType.NO_CACHE,
             check=False,
             check_fail_continue=False,
             init_hook=None,
             **outter_kwargs):
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
                        It could be a list of InputType object. For example,
                        input_types=[dense_vector(9), integer_value(2)]. Or user
                        can set a dict of InputType object, which key is
                        data_layer's name. For example, input_types=\
                        {'img': img_features, 'label': label}. when using dict of
                        InputType, user could yield a dict of feature values, which
                        key is also data_layer's name.

    :type input_types: list|tuple|dict

    :param should_shuffle: True if data should shuffle. Pass None means shuffle
                           when is training and not to shuffle when is testing.
    :type should_shuffle: bool

    :param pool_size: Max number of sample in data pool.
    :type pool_size: int

    :param min_pool_size: Set minimal sample in data pool. The PaddlePaddle will
                          random pick sample in pool. So the min_pool_size
                          effect the randomize of data.
    :type min_pool_size: int

    :param can_over_batch_size: True if paddle can return a mini-batch larger
                                than batch size in settings. It is useful when
                                custom calculate one sample's batch_size.

                                It is very danger to set it to false and use
                                calc_batch_size together. Default is true.
    :type can_over_batch_size: bool

    :param calc_batch_size: a method to calculate each sample's batch size.
                            Default each sample's batch size is 1. But to you
                            can customize each sample's batch size.
    :type calc_batch_size: callable

    :param cache: Cache strategy of Data Provider. Default is CacheType.NO_CACHE
    :type cache: int

    :param init_hook: Initialize hook. Useful when data provider need load some
                      external data like dictionary. The parameter is
                      (settings, file_list, \*\*kwargs).

                      - settings. It is the global settings object. User can set
                        settings.input_types here.
                      - file_list. All file names for passed to data provider.
                      - is_train. Is this data provider used for training or not.
                      - kwargs. Other keyword arguments passed from
                        trainer_config's args parameter.
    :type init_hook: callable

    :param check: Check the yield data format is as same as input_types. Enable
                  this will make data provide process slow but it is very useful
                  for debug. Default is disabled.
    :type check: bool

    :param check_fail_continue: Continue train or not when check failed. Just
                                drop the wrong format data when it is True. Has
                                no effect when check set to False.
    :type check_fail_continue: bool
    """

    def __wrapper__(generator):
        class DataProvider(object):
            def __init__(self, file_list, **kwargs):
                self.logger = logging.getLogger("")
                self.logger.setLevel(logging.INFO)
                self.input_types = None
                self.should_shuffle = should_shuffle

                true_table = [1, 't', 'true', 'on']
                false_table = [0, 'f', 'false', 'off']
                if not isinstance(self.should_shuffle, bool) and \
                                self.should_shuffle is not None:

                    if isinstance(self.should_shuffle, basestring):
                        self.should_shuffle = self.should_shuffle.lower()

                    if self.should_shuffle in true_table:
                        self.should_shuffle = True
                    elif self.should_shuffle in false_table:
                        self.should_shuffle = False
                    else:
                        self.logger.warning(
                            "Could not recognize should_shuffle (%s), "
                            "just use default value of should_shuffle."
                            " Please set should_shuffle to bool value or "
                            "something in %s" %
                            (repr(self.should_shuffle),
                             repr(true_table + false_table)))
                        self.should_shuffle = None

                self.pool_size = pool_size
                self.can_over_batch_size = can_over_batch_size
                self.calc_batch_size = calc_batch_size
                self.file_list = file_list
                self.generator = generator
                self.cache = cache
                self.min_pool_size = min_pool_size
                self.input_order = kwargs['input_order']
                self.check = check
                if init_hook is not None:
                    init_hook(self, file_list=file_list, **kwargs)

                if 'slots' in outter_kwargs:
                    self.logger.warning('setting slots value is deprecated, '
                                        'please use input_types instead.')
                    self.slots = outter_kwargs['slots']
                if input_types is not None:
                    self.slots = input_types

                if self.input_types is not None:
                    self.slots = self.input_types

                assert self.slots is not None, \
                    "Data Provider's input_types must be set"
                assert self.generator is not None

                use_dynamic_order = False
                if isinstance(self.slots, dict):  # reorder input_types
                    self.slots = [self.slots[ipt] for ipt in self.input_order]
                    use_dynamic_order = True

                if len(self.slots) == 1:
                    self.generator = SingleSlotWrapper(self.generator)

                if use_dynamic_order:
                    self.generator = InputOrderWrapper(self.generator,
                                                       self.input_order)
                else:
                    self.generator = CheckInputTypeWrapper(
                        self.generator, self.slots, self.logger)
                if self.check:
                    self.generator = CheckWrapper(self.generator, self.slots,
                                                  check_fail_continue,
                                                  self.logger)

        return DataProvider

    return __wrapper__


def deserialize_args(args):
    """
    Internal use only.
    :param args:
    :return:
    """
    return cPickle.loads(args)
