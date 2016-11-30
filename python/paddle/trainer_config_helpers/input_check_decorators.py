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
"""
Input Check Decorators. The decorators and utilities related to check the
layer inputs are legal.  The main API is check_input. Please reference
documentation in check_input for detail usages.
"""

from paddle.trainer.PyDataProvider2 import InputType, SequenceType, DataType
import functools
import collections

__all__ = [
    "check_input", "AcceptInput", "SameSeqType", "OutputSize", "OutputType",
    "InputSize", "SameOutputType", "CompositeChecker",
    "default_seq_type_and_size"
]


def base_check_input(callback):
    """
    A layer decorator.
    Using the callback to set layer's output type. The callback method takes the
    input types and this layer output as the parameter and returns this layer's
    type. The callback type should be ([InputType], LayerOutput) => InputType.

    :param callback: a function set current layer's type, with type
                     ([InputType], LayerOutput) => InputType.
    :type callback: callable
    :return: wrapped method.
    :rtype: callable
    """

    def __impl__(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            parent_types = map(lambda x: x.input_type, output.parents)
            for each_parent_type in parent_types:
                assert each_parent_type is None or isinstance(each_parent_type,
                                                              InputType)
            output.input_type = callback(parent_types, output)
            return output

        return wrapper

    return __impl__


class InputSize(object):
    """
    Check the number of inputs in the current layer of the neural network equals
    :code:`input_number` or within :code:`input_number`

    :param input_number: the exact number, or list, of inputs in the current
                         layer.
    :type input_number: int|list|tuple
    :rtype: callable
    """

    def __init__(self, input_number):
        self.input_number = input_number
        if not isinstance(input_number, collections.Sequence):
            self.input_number = [self.input_number]
        assert isinstance(self.input_number, collections.Sequence)

        for each in self.input_number:
            assert isinstance(each, int)

    def __call__(self, parent_types, output, next_callback):
        assert len(parent_types) in self.input_number
        return next_callback(parent_types, output)


class AcceptInput(object):
    """
    Check the inputs of the current layer should be some data_type or seq_type.

    :param idx: Check the input with index `idx`. If idx = -1, then check all
                input with same constraints.
    :type idx: int
    :param data_type: It could be a list of data types or the exact data type
                      that input should be. None will be accept any data type.
    :type data_type: int|list
    :param seq_type: It could be a list of sequence types or the exact sequence
                     type that input should be. None will be accept any sequence
                     type.
    :type seq_type: int|list
    :rtype: callable
    """

    def __init__(self, idx=-1, data_type=None, seq_type=None):
        self.idx = idx
        self.data_type = data_type
        if self.data_type is not None:
            if not isinstance(self.data_type, collections.Sequence):
                self.data_type = [self.data_type]
            assert isinstance(self.data_type, collections.Sequence)
            for each_data_type in self.data_type:
                assert DataType.is_valid(each_data_type)

        self.seq_type = seq_type
        if self.seq_type is not None:
            if not isinstance(self.seq_type, collections.Sequence):
                self.seq_type = [self.seq_type]

            for each_seq_type in self.seq_type:
                assert SequenceType.is_valid(each_seq_type)

    def __call__(self, parent_types, output, next_callback):
        for idx, tp in enumerate(parent_types):
            if self.idx != -1 and self.idx != idx:
                continue
            assert isinstance(tp, InputType)

            if self.data_type is not None:
                assert tp.type in self.data_type

            if self.seq_type is not None:
                assert tp.seq_type in self.seq_type

        return next_callback(parent_types, output)


class SameSeqType(object):
    """
    Set the output sequence type of current layer as same as the input sequence
    type. The each input of current layer should have same sequence type.
    """

    def __init__(self):
        pass

    def __call__(self, parent_types, output, next_callback):
        seq_types = [pt.seq_type for pt in parent_types]
        all_input_sequence_same = len(
            filter(lambda x: x != seq_types[0], seq_types)) == 0
        assert all_input_sequence_same
        tp = next_callback(parent_types, output)
        assert isinstance(tp, InputType)
        tp.seq_type = seq_types[0]
        return tp


class OutputSize(object):
    """
    Set output size as the LayerOutput.size
    """

    def __init__(self):
        pass

    def __call__(self, parent_types, output, next_callback):
        tp = next_callback(parent_types, output)
        tp.dim = output.size
        return tp


class CompositeChecker(object):
    """
    Composite several checker to one checker.
    """

    def __init__(self, *callbacks):
        self.reversed_callbacks = reversed(callbacks)

    def __call__(self, parent_types, output, next_callback):
        for each in self.reversed_callbacks:
            next_callback = functools.partial(each, next_callback=next_callback)

        return next_callback(parent_types, output)


class OutputType(object):
    """
    Set output data type as data_type.
    """

    def __init__(self, data_type):
        self.data_type = data_type
        assert DataType.is_valid(self.data_type)

    def __call__(self, parent_types, output, next_callback):
        tp = next_callback(parent_types, output)
        assert isinstance(tp, InputType)
        tp.type = self.data_type
        return tp


class SameOutputType(object):
    """
    Set output data type as the input data type.
    """

    def __init__(self):
        pass

    def __call__(self, parent_types, output, next_callback):
        data_types = [tp.type for tp in parent_types]
        assert len(filter(lambda x: x != data_types[0], data_types)) == 0
        data_type = data_types[0]
        tp = next_callback(parent_types, output)
        tp.type = data_type
        return tp


def default_seq_type_and_size():
    """
    Set the output of the current layer use same sequence type as input, and set
    the output size as same as LayerOutput.size.
    """
    return CompositeChecker(SameSeqType(), OutputSize())


def basic_checker(*args, **kwargs):
    """
    The inner most callback in middleware chain. Just return a InputType.
    """
    return InputType(0, 0, 0)


def check_input(*callbacks):
    """
    A layer decorator. Check inputs of this layer fit all callbacks.

    If any input type of current layer is None, then this layer's output type is
    None. Otherwise, invoke all callbacks one by one. The callbacks are
    middleware composed in a chain. Every callback should be a method with type
    (parent_types:[InputType], output:LayerOutput,
    next_callback:([InputType], output)=>InputType)=>InputType, and the
    next_callback should be invoked in every callback. The callback will return
    the InputType which is modified by the next_callback return value. Every
    callback could throw an AssertError when the input is not legal.

    :param callbacks: middleware callbacks. with type
                      ([InputType], LayerOutput, ([InputType],
                          LayerOutput)=>InputType) => InputType.
    :type callbacks: tuple of callable
    :return: wrapped method
    :rtype: callable
    """

    def callback_impl(parent_types, output):
        for each_parent_type in parent_types:
            if each_parent_type is None:
                return None
        checker = CompositeChecker(*callbacks)
        return checker(parent_types, output, basic_checker)

    return base_check_input(callback_impl)
