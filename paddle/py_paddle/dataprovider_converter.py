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

import paddle.trainer.PyDataProvider2 as dp2
import collections
import swig_paddle
import numpy

__all__ = ['DataProviderConverter']


class IScanner(object):
    def __init__(self, input_type, pos):
        self.input_type = input_type
        assert isinstance(self.input_type, dp2.InputType)
        self.pos = pos

    def scan(self, dat):
        pass

    def finish_scan(self, argument):
        pass


class DenseScanner(IScanner):
    """
    :type __mat__: numpy.ndarray
    """

    def __init__(self, input_type, pos):
        IScanner.__init__(self, input_type, pos)
        self.__mat__ = None

    def scan(self, dat):
        if self.__mat__ is None:
            self.__mat__ = numpy.array([dat], dtype='float32')
        else:
            self.__mat__ = numpy.append(self.__mat__, [dat], axis=0)

    def finish_scan(self, argument):
        assert isinstance(argument, swig_paddle.Arguments)
        assert isinstance(self.input_type, dp2.InputType)
        if self.__mat__.dtype != numpy.float32:
            self.__mat__ = self.__mat__.astype(numpy.float32)
        m = swig_paddle.Matrix.createDenseFromNumpy(self.__mat__, True, False)
        argument.setSlotValue(self.pos, m)


class SparseBinaryScanner(IScanner):
    def __init__(self, input_type, pos):
        IScanner.__init__(self, input_type, pos)
        self.__rows__ = [0]
        self.__cols__ = []
        self.__height__ = 0
        self.__nnz__ = 0
        self.__value__ = []

    def scan(self, dat):
        self.extend_cols(dat)
        self.__rows__.append(len(self.__cols__))
        self.__height__ += 1

    def extend_cols(self, dat):
        self.__cols__.extend(dat)

    def finish_scan(self, argument):
        assert isinstance(argument, swig_paddle.Arguments)
        assert isinstance(self.input_type, dp2.InputType)
        m = swig_paddle.Matrix.createSparse(self.__height__,
                                            self.input_type.dim,
                                            len(self.__cols__),
                                            len(self.__value__) == 0)
        assert isinstance(m, swig_paddle.Matrix)
        m.sparseCopyFrom(self.__rows__, self.__cols__, self.__value__)
        argument.setSlotValue(self.pos, m)


class SparseFloatScanner(SparseBinaryScanner):
    def __init__(self, input_type, pos):
        SparseBinaryScanner.__init__(self, input_type, pos)

    def extend_cols(self, dat):
        self.__cols__.extend((x[0] for x in dat))
        self.__value__.extend((x[1] for x in dat))


class IndexScanner(IScanner):
    def __init__(self, input_type, pos):
        IScanner.__init__(self, input_type, pos)
        self.__ids__ = []

    def scan(self, dat):
        self.__ids__.append(dat)

    def finish_scan(self, argument):
        ids = swig_paddle.IVector.create(self.__ids__)
        assert isinstance(argument, swig_paddle.Arguments)
        argument.setSlotIds(self.pos, ids)


class SequenceScanner(IScanner):
    def __init__(self, input_type, pos, inner_scanner, setter):
        IScanner.__init__(self, input_type, pos)
        self.__seq__ = [0]
        self.__inner_scanner__ = inner_scanner
        self.__setter__ = setter

    def scan(self, dat):
        self.__seq__.append(self.__seq__[-1] + self.get_size(dat))
        for each in dat:
            self.__inner_scanner__.scan(each)

    def finish_scan(self, argument):
        seq = swig_paddle.IVector.create(self.__seq__, False)
        self.__setter__(argument, self.pos, seq)
        self.__inner_scanner__.finish_scan(argument)

    def get_size(self, dat):
        if isinstance(self.__inner_scanner__, SequenceScanner):
            return sum(self.__inner_scanner__.get_size(item) for item in dat)
        else:
            return len(dat)


class DataProviderConverter(object):
    def __init__(self, input_types):
        self.input_types = input_types
        assert isinstance(self.input_types, collections.Sequence)
        for each in self.input_types:
            assert isinstance(each, dp2.InputType)

    def convert(self, dat, argument=None):
        if argument is None:
            argument = swig_paddle.Arguments.createArguments(0)
        assert isinstance(argument, swig_paddle.Arguments)
        argument.resize(len(self.input_types))

        scanners = [
            DataProviderConverter.create_scanner(i, each_type)
            for i, each_type in enumerate(self.input_types)
        ]

        for each_sample in dat:
            for each_step, scanner in zip(each_sample, scanners):
                scanner.scan(each_step)

        for scanner in scanners:
            scanner.finish_scan(argument)

        return argument

    def __call__(self, dat, argument=None):
        return self.convert(dat, argument)

    @staticmethod
    def create_scanner(i, each):
        assert isinstance(each, dp2.InputType)
        retv = None
        if each.type == dp2.DataType.Dense:
            retv = DenseScanner(each, i)
        elif each.type == dp2.DataType.Index:
            retv = IndexScanner(each, i)
        elif each.type == dp2.DataType.SparseNonValue:
            retv = SparseBinaryScanner(each, i)
        elif each.type == dp2.DataType.SparseValue:
            retv = SparseFloatScanner(each, i)
        assert retv is not None

        if each.seq_type == dp2.SequenceType.SUB_SEQUENCE:
            retv = SequenceScanner(
                each, i, retv,
                lambda a, p, seq: a.setSlotSubSequenceStartPositions(p, seq))

        if each.seq_type in [
                dp2.SequenceType.SUB_SEQUENCE, dp2.SequenceType.SEQUENCE
        ]:
            retv = SequenceScanner(
                each, i, retv,
                lambda a, p, seq: a.setSlotSequenceStartPositions(p, seq))
        return retv
