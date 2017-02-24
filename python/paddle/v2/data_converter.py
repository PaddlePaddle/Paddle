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

import collections
import py_paddle.swig_paddle
import numpy
import paddle.trainer.PyDataProvider2 as dp2

__all__ = ['DataConverter']


class IDataConverter(object):
    def __init__(self, input_type, pos):
        """
        :param input_type: data type
        :type input_type: dp2.InputType
        :param pos: which input, start from 0
        :type pos: int
        """
        self.input_type = input_type
        assert isinstance(self.input_type, dp2.InputType)
        self.pos = pos

    def convert(self, data, argument):
        """
        Conv data to paddle format.
        :param data: input data
        :param argument: paddle format
        """
        pass


class DenseConvert(IDataConverter):
    def __init__(self, input_type, pos):
        IDataConverter.__init__(self, input_type, pos)

    def convert(self, data, argument):
        """
        :param data: input data
        :type data: list | numpy array
        :param argument: the type which paddle is acceptable
        :type argument: swig_paddle.Arguments
        """
        assert isinstance(argument, swig_paddle.Arguments)
        if data.dtype != numpy.float32:
            data = data.astype(numpy.float32)
        m = swig_paddle.Matrix.createDenseFromNumpy(data, True, False)
        argument.setSlotValue(self.pos, m)


class SparseBinaryConvert(IDataConverter):
    def __init__(self, input_type, pos):
        IDataConverter.__init__(self, input_type, pos)
        self.__rows__ = [0]
        self.__cols__ = []
        self.__height__ = 0
        self.__nnz__ = 0
        self.__value__ = []

    def fill_csr(self, data):
        self.__height__ = len(data)
        for x in data:
            self.__rows__.append(self.__rows__[-1] + len(x))
        self__cols__ = data.flatten()

    def convert(self, data, argument):
        assert isinstance(argument, swig_paddle.Arguments)

        fill_csr(data)
        m = swig_paddle.Matrix.createSparse(self.__height__,
                                            self.input_type.dim,
                                            len(self.__cols__),
                                            len(self.__value__) == 0)
        assert isinstance(m, swig_paddle.Matrix)
        m.sparseCopyFrom(self.__rows__, self.__cols__, self.__value__)
        argument.setSlotValue(self.pos, m)


class SparseFloatConvert(SparseBinaryConvert):
    def __init__(self, input_type, pos):
        SparseBinaryConvert.__init__(self, input_type, pos)

    def fill_csr(self, data):
        self.__height__ = len(data)
        for x in data:
            self.__rows__.append(self.__rows__[-1] + len(x))
        self.__cols__.extend((x[0] for x in data))
        self.__value__.extend((x[1] for x in data))


class IndexConvert(IDataConverter):
    def __init__(self, input_type, pos):
        IDataConverter.__init__(self, input_type, pos)
        self.__ids__ = []

    def convert(self, data, argument):
        assert isinstance(argument, swig_paddle.Arguments)
        self.__ids__ = data.flatten()
        ids = swig_paddle.IVector.create(self.__ids__)
        argument.setSlotIds(self.pos, ids)


class SequenceConvert(IDataConverter):
    def __init__(self, input_type, pos, inner_convert, setter):
        """
        :param input_type: the type of input data
        :type input_type: dp2.InputType
        :param pos: the position of this input
        :type pos: int
        :param inner_convert: DataConvert type
        :type inner_convert: DenseConvert|SparseBinaryConvert|
                             SparseFloatConvert|IndexConvert
        :param setter:
        :type setter:
        """
        IDataConverter.__init__(self, input_type, pos)
        self.__seq__ = [0]
        self.__inner_convert__ = inner_convert
        self.__setter__ = setter

    def fill_seq(self, data):
        for each in data:
            self.__seq__.append(self.__seq__[-1] + self.get_size(each))

    def convert(self, data, argument):
        fill_seq(data)
        seq = swig_paddle.IVector.create(self.__seq__, False)
        self.__setter__(argument, self.pos, seq)

        dat = []
        for each in data:
            dat.append(each)
        self.__inner_scanner__.convert(dat, argument)

    def get_size(self, data):
        if isinstance(self.__inner_scanner__, SequenceConvert):
            return sum(self.__inner_scanner__.get_size(item) for item in dat)
        else:
            return len(data)


class DataConverter(object):
    def __init__(self, input_mapper):
        """
        Usege:

        .. code-block:: python
            inputs = [('image', dense_vector), ('label', integer_value)]
            cvt = DataConverter(inputs)
            arg = cvt.convert(minibatch_data, {'image':0, 'label':1})

        :param input_mapper: list of (input_name, input_type)
        :type input_mapper: list
        """
        assert isinstance(self.input_types, collections.Sequence)
        self.input_names = []
        self.input_types = []
        for each in self.input_types:
            self.input_names.append(each[0])
            self.input_types.append(each[1])
            assert isinstance(each[1], dp2.InputType)

    def convert(self, data, input_dict=None, argument=None):
        """
        Convert minibatch data to Paddle's argument. The data is numpy array
        or list.

        :param data: input samples, for example, [column0, column1, ...] or
                     (column0, column1, ...) each column is one minibatch
                     feature. Note, if only one column featrue, data also
                     shuld be a list or tupe, [column0] or (column0).
        :type data: list|tuple
        :param input_dict: a dictionary to specify the correspondence
                           of data_layer and input data. If None,
                           the feature order in argument and data is the same.
        :type input_dict: dict, like {string:integer, string, integer, ...}|None
        :param argument: converted data will be saved in this argument. If None,
                         it will create a swig_paddle.Arguments firstly.
        :param type: swig_paddle.Arguments|None
        """
        if argument is None:
            argument = swig_paddle.Arguments.createArguments(0)
        assert isinstance(argument, swig_paddle.Arguments)
        argument.resize(len(self.input_types))

        converts = [
            DataConverter.create_scanner(i, each_type)
            for i, each_type in enumerate(self.input_types)
        ]

        for i, cvt in enumerate(converts):
            if input_dict is not None:
                dat = data[input_dict[self.input_names[i]]]
            else:
                dat = data[i]
            cvt.convert(dat, argument)

        return argument

    def __call__(self, dat, argument=None):
        return self.convert(dat, argument)

    @staticmethod
    def create_scanner(pos, each):
        assert isinstance(each, dp2.InputType)
        retv = None
        if each.type == dp2.DataType.Dense:
            retv = DenseConvert(each, pos)
        elif each.type == dp2.DataType.Index:
            retv = IndexConvert(each, pos)
        elif each.type == dp2.DataType.SparseNonValue:
            retv = SparseBinaryConvert(each, pos)
        elif each.type == dp2.DataType.SparseValue:
            retv = SparseFloatConvert(each, pos)
        assert retv is not None

        if each.seq_type == dp2.SequenceType.SUB_SEQUENCE:
            retv = SequenceConvert(
                each, pos, retv,
                lambda arg, pos, seq: arg.setSlotSubSequenceStartPositions(pos, seq)
            )

        if each.seq_type in [
                dp2.SequenceType.SUB_SEQUENCE, dp2.SequenceType.SEQUENCE
        ]:
            retv = SequenceConvert(
                each, pos, retv,
                lambda arg, pos, seq: arg.setSlotSequenceStartPositions(pos, seq)
            )
        return retv
