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
import itertools
from functools import reduce

__all__ = ['DataProviderConverter']


class IScanner(object):
    """
    The scanner will scan Python object two passes, then convert it to Paddle's
    argument.

    In the first pass, `pre_scan` will be invoked by every data instance, and
    then invoke `finish_pre_scan` to arguments. And the second pass do the same
    thing except the functions changed to `scan`, `finish_scan`.

    During the first pass, a scanner may count the shape of input matrix and
    allocate memory for this argument. Then fill the data into this  argument
    in second pass.
    """

    def __init__(self, input_type, pos):
        self.input_type = input_type
        if not isinstance(self.input_type, dp2.InputType):
            raise ValueError("input type should be dataprovider2.InputType")
        self.pos = pos
        # data_in_gpu is used to indicate whether to create argument on GPU
        # or not in GPU mode. Now if using one thread (trainer_count=1),
        # trainer uses NeuralNetwork which needs to create argument on GPU
        # before calling forward function. So, set data_in_gpu to True.
        # Otherwise, trainer uses MultiGradientMachine which will transfer
        # data from CPU to GPU in the forward function, set data_in_gpu to
        # False in this case.
        self.data_in_gpu = swig_paddle.isUsingGpu(
        ) and swig_paddle.getTrainerCount() == 1

    def pre_scan(self, dat):
        """
        First pass scan method. During this method, the scanner could count the
        data number, and get the total memory size this batch would use.

        :param dat: The python object.
        """
        pass

    def finish_pre_scan(self, argument):
        """
        Finish first scan pass. Allocate the memory.

        :param argument: Output arguments object.
        :type argument: swig_paddle.Arguments
        :param dat: Output arguments object.
        :type dat: The Python object, numpy.array or List.
        :return:
        """
        pass

    def scan(self, dat):
        """
        Second pass scan method. Copy the data to arguments.

        :param dat: The python object.
        """
        pass

    def finish_scan(self, argument):
        """
        Finish second pass. Finalize the resources, etc.

        :param argument: Output arguments object.
        :type argument: swig_paddle.Arguments
        """
        pass


class DenseScanner(IScanner):
    """
    :type __mat__: numpy.ndarray
    """

    def __init__(self, input_type, pos):
        IScanner.__init__(self, input_type, pos)
        self.__mat__ = None
        self.__shape__ = None
        self.__height__ = 0
        self.__dim__ = 0

    def pre_scan(self, dat):
        self.__height__ += 1
        if self.__shape__ is None:
            self.__shape__ = numpy.array(dat).shape
            if len(self.__shape__) > 3:
                raise ValueError(
                    "The dimension of input cannot be greater than 3.")
            if len(self.__shape__) == 0:
                raise ValueError(
                    "The input should be a vector, please check your input data."
                )
            self.__dim__ = reduce(lambda x, y: x * y, self.__shape__)
            if len(self.__shape__) == 1 and self.__dim__ != self.input_type.dim:
                raise ValueError(
                    "The data size must be equal to it in data layer.")
        else:
            if self.__shape__ != numpy.array(dat).shape:
                raise ValueError(
                    "The data shape must be same in one mini-batch.")

    def finish_pre_scan(self, argument):
        self.__mat__ = numpy.ndarray(
            shape=(self.__height__, self.__dim__), dtype=numpy.float32)
        self.__height__ = 0

    def scan(self, dat):
        # It's better to use NumPy array for speed.
        dat = numpy.array(dat)
        dat = dat.flatten()
        self.__mat__[self.__height__] = dat
        self.__height__ += 1

    def finish_scan(self, argument):
        assert isinstance(argument, swig_paddle.Arguments)
        if self.__mat__.dtype != numpy.float32:
            self.__mat__ = self.__mat__.astype(numpy.float32)
        m = swig_paddle.Matrix.createDenseFromNumpy(self.__mat__, True,
                                                    self.data_in_gpu)
        argument.setSlotValue(self.pos, m)
        if len(self.__shape__) > 1:
            # The last-two dimenstions are the frame height and width.
            # For example, the layout is CHW for 3-D feature of image.
            # The H and W are the frame height and width.
            h, w = self.__shape__[-2:]
            argument.setSlotFrameHeight(self.pos, h)
            argument.setSlotFrameWidth(self.pos, w)
        self.__shape__ = None


class SparseBinaryScanner(IScanner):
    def __init__(self, input_type, pos):
        IScanner.__init__(self, input_type, pos)
        self.__rows__ = [0]
        self.__cols__ = []
        self.__height__ = 0
        self.__value__ = []

    def scan(self, dat):
        self.extend_cols(dat)
        self.__rows__.append(len(self.__cols__))
        self.__height__ += 1

    def extend_cols(self, dat):
        self.__cols__.extend(dat)

    def finish_scan(self, argument):
        assert isinstance(argument, swig_paddle.Arguments)
        m = swig_paddle.Matrix.createSparse(
            self.__height__,
            self.input_type.dim,
            len(self.__cols__),
            len(self.__value__) == 0,
            False,  # trans
            False)  # TODO supoort GPU
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
        self.__ids__ = None
        self.__idx__ = 0

    def pre_scan(self, dat):
        self.__idx__ += 1

    def finish_pre_scan(self, argument):
        self.__ids__ = [0] * self.__idx__
        self.__idx__ = 0

    def scan(self, dat):
        self.__ids__[self.__idx__] = dat
        self.__idx__ += 1

    def finish_scan(self, argument):
        ids = swig_paddle.IVector.create(self.__ids__, self.data_in_gpu)
        assert isinstance(argument, swig_paddle.Arguments)
        argument.setSlotIds(self.pos, ids)


class SequenceScanner(IScanner):
    def __init__(self, input_type, pos, inner_scanner, setter):
        IScanner.__init__(self, input_type, pos)
        self.__seq__ = [0]
        self.__inner_scanner__ = inner_scanner
        self.__setter__ = setter

    def pre_scan(self, dat):
        for each in dat:
            self.__inner_scanner__.pre_scan(each)

    def finish_pre_scan(self, argument):
        self.__inner_scanner__.finish_pre_scan(argument)

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
            for each_step, scanner in itertools.izip(each_sample, scanners):
                scanner.pre_scan(each_step)

        for scanner in scanners:
            scanner.finish_pre_scan(argument)

        for each_sample in dat:
            for each_step, scanner in itertools.izip(each_sample, scanners):
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
