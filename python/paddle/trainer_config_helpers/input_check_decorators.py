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

from paddle.trainer.PyDataProvider2 import InputType, SequenceType, DataType
import functools
import collections

__all__ = [
    "input_mapping", "AcceptInput", "SameSeqType", "SameOutputDim", "OutputType"
]


def base_input_mapping(callback):
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


class AcceptInput(object):
    def __init__(self, idx=-1, data_type=None, seq_type=None):
        self.idx = idx
        self.data_type = data_type
        if self.data_type is not None:
            if not isinstance(self.data_type, collections.Sequence):
                self.data_type = [self.data_type]
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


class NewInputType(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return InputType(0, 0, 0)


class SameSeqType(object):
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


class SameOutputDim(object):
    def __init__(self):
        pass

    def __call__(self, parent_types, output, next_callback):
        tp = next_callback(parent_types, output)
        tp.dim = output.size
        return tp


class OutputType(object):
    def __init__(self, data_type):
        self.data_type = data_type
        assert DataType.is_valid(self.data_type)

    def __call__(self, parent_types, output, next_callback):
        tp = next_callback(parent_types, output)
        assert isinstance(tp, InputType)
        tp.type = self.data_type
        return tp


def input_mapping(*callbacks):
    def callback_impl(parent_types, output):
        for each_parent_type in parent_types:
            if each_parent_type is None:
                return None
        base = NewInputType()

        for callback in reversed(callbacks):
            base = lambda ptypes, opts: callback(ptypes, opts, base)

        return base(parent_types, output)

    return base_input_mapping(callback_impl)
