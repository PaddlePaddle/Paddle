#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import struct
from typing import TYPE_CHECKING

import numpy as np

import paddle
from paddle import pir

from ..pir import Value
from ..pir.core import _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE, ParameterMeta
from . import core
from .framework import (
    Variable,
    _cpu_num,
    _cuda_ids,
    default_main_program,
    in_dygraph_mode,
    in_pir_mode,
)

if TYPE_CHECKING:
    from paddle._typing import DTypeLike
    from paddle._typing.dtype_like import _DTypeLiteral

__all__ = []

_PADDLE_DTYPE_2_NUMPY_DTYPE = {
    core.VarDesc.VarType.BOOL: 'bool',
    core.VarDesc.VarType.FP8_E4M3FN: 'float8_e4m3fn',
    core.VarDesc.VarType.FP8_E5M2: 'float8_e5m2',
    core.VarDesc.VarType.FP16: 'float16',
    core.VarDesc.VarType.BF16: 'uint16',
    core.VarDesc.VarType.FP32: 'float32',
    core.VarDesc.VarType.FP64: 'float64',
    core.VarDesc.VarType.INT8: 'int8',
    core.VarDesc.VarType.INT16: 'int16',
    core.VarDesc.VarType.INT32: 'int32',
    core.VarDesc.VarType.INT64: 'int64',
    core.VarDesc.VarType.UINT8: 'uint8',
    core.VarDesc.VarType.COMPLEX64: 'complex64',
    core.VarDesc.VarType.COMPLEX128: 'complex128',
}

_NUMPY_DTYPE_2_PADDLE_DTYPE = {
    'bool': core.VarDesc.VarType.BOOL,
    'float16': core.VarDesc.VarType.FP16,
    'uint16': core.VarDesc.VarType.BF16,
    'float32': core.VarDesc.VarType.FP32,
    'float64': core.VarDesc.VarType.FP64,
    'int8': core.VarDesc.VarType.INT8,
    'int16': core.VarDesc.VarType.INT16,
    'int32': core.VarDesc.VarType.INT32,
    'int64': core.VarDesc.VarType.INT64,
    'uint8': core.VarDesc.VarType.UINT8,
    'complex64': core.VarDesc.VarType.COMPLEX64,
    'complex128': core.VarDesc.VarType.COMPLEX128,
}


def convert_float_to_uint16(data, data_format="NCHW"):
    if data.size == 0:
        return data.view(np.uint16)

    if data_format == "NHWC":
        data = np.transpose(data, [0, 3, 1, 2])

    new_data = np.vectorize(
        lambda x: struct.unpack('<I', struct.pack('<f', x))[0] >> 16,
        otypes=[np.uint16],
    )(data.flat)
    new_data = np.reshape(new_data, data.shape)

    if data_format == "NHWC":
        new_data = np.transpose(new_data, [0, 2, 3, 1])
    return new_data


def convert_uint16_to_float(data):
    new_data = np.vectorize(
        lambda x: struct.unpack('<f', struct.pack('<I', x << 16))[0],
        otypes=[np.float32],
    )(data.flat)
    return np.reshape(new_data, data.shape)


def convert_dtype(dtype: DTypeLike) -> _DTypeLiteral:
    if isinstance(dtype, core.VarDesc.VarType):
        if dtype in _PADDLE_DTYPE_2_NUMPY_DTYPE:
            return _PADDLE_DTYPE_2_NUMPY_DTYPE[dtype]
    if isinstance(dtype, core.DataType):
        if dtype in _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE:
            return _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[dtype]
    elif isinstance(dtype, type):
        # This branch is for NumPy scalar types
        if dtype in [
            bool,
            np.float16,
            np.uint16,
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.complex64,
            np.complex128,
        ]:
            return dtype.__name__
    else:
        # This branch is for np.dtype and str
        if dtype in [
            'bool',
            'float16',
            'uint16',
            'float32',
            'float64',
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
            'complex64',
            'complex128',
            'float8_e4m3fn',
            'float8_e5m2',
        ]:
            # NOTE(SigureMo): Since the np.dtype object is not an instance of
            # type, so it will not be handled by the previous branch. We need
            # to convert it to str here.
            return str(dtype)
        # NOTE(zhangbo): Now numpy does not support bfloat, so use numpy.uint16 to represent paddle.bfloat16, there binaries are consistent.
        # If cast ndarray to uint16 and trans to tensor, should not ndarray.astype('uint16') directly
        # should use function 'convert_float_to_uint16' above, otherwise bits is wrong
        if dtype in ['bfloat16']:
            return 'uint16'

    raise TypeError(
        "dtype must be any of [bool, float16, uint16, float32, float64, int8, int16, "
        f"int32, int64, uint8, complex64, complex128, bfloat16], but received {dtype}"
    )


def check_variable_and_dtype(
    input, input_name, expected_dtype, op_name, extra_message=''
):
    if in_pir_mode():
        check_type(
            input, input_name, (Value, ParameterMeta), op_name, extra_message
        )
    else:
        check_type(input, input_name, (Variable, Value), op_name, extra_message)
    check_dtype(input.dtype, input_name, expected_dtype, op_name, extra_message)


def check_type(input, input_name, expected_type, op_name, extra_message=''):
    # NOTE [ Why skip dynamic graph check ]:
    # 1. If the input type / dtype of a layer is wrong, it will be reported
    # directly on that line. User can easily print the relevant information
    # on which line. It is easier to debug, so there is no need to check
    # in dynamic graph mode.
    # 2. Performance considerations. Because these checks are executed at
    # each step in dynamic graph mode, it will bring a heavy performance burden.
    if in_dygraph_mode():
        return

    # NOTE: `in_to_static_mode` is used to determined whether this op is called under
    # @to_static in transformation from dygraph to static layer. We add Tensor in
    # expected_type to skip checking because Tensor may be created and used in unusual way.
    from .dygraph.base import in_to_static_mode

    # Need a better design to be fix this.
    if in_to_static_mode():
        if not isinstance(expected_type, tuple):
            expected_type = (expected_type,)
        expected_type += (core.eager.Tensor,)
    elif isinstance(input, core.eager.Tensor):
        raise TypeError(
            "Please use `with base.dygraph.guard()` as context or `base.enable_dygraph()` to switch to imperative mode firstly. "
            f"Because received '{input_name}' in {op_name} is a imperative Variable."
        )
    if not isinstance(input, expected_type):
        raise TypeError(
            f"The type of '{input_name}' in {op_name} must be {expected_type}, but received {type(input)}. {extra_message}"
        )


def check_dtype(
    input_dtype, input_name, expected_dtype, op_name, extra_message=''
):
    # See NOTE [ Why skip dynamic graph check ]
    if in_dygraph_mode():
        return

    if convert_dtype(input_dtype) not in expected_dtype:
        raise TypeError(
            f"The data type of '{input_name}' in {op_name} must be {expected_dtype}, but received {convert_dtype(input_dtype)}. {extra_message}"
        )


def check_shape(
    shape,
    op_name,
    expected_shape_type=(list, tuple, Variable, Value),
    expected_element_type=(int, Variable, Value),
    expected_tensor_dtype=('int32', 'int64'),
):
    # See NOTE [ Why skip dynamic graph check ]
    if in_dygraph_mode():
        return
    check_type(shape, 'shape', expected_shape_type, op_name)
    if expected_element_type is not None and not isinstance(
        shape, (Variable, Value)
    ):
        for item in shape:
            check_type(item, 'element of shape', expected_element_type, op_name)
            if expected_tensor_dtype is not None and isinstance(
                item, (Variable, Value)
            ):
                check_dtype(
                    item.dtype,
                    'element of shape',
                    expected_tensor_dtype,
                    op_name,
                    'If element of shape is Tensor, its data type should be {}'.format(
                        ', '.join(expected_tensor_dtype)
                    ),
                )
    if expected_tensor_dtype is not None and isinstance(
        shape, (Variable, Value)
    ):
        check_dtype(shape.dtype, 'shape', expected_tensor_dtype, op_name)


class DataToLoDTensorConverter:
    def __init__(self, place, lod_level, shape, dtype):
        self.place = place
        self.lod_level = lod_level
        self.shape = shape
        negative_count = 0
        for s in self.shape:
            if s < 0:
                negative_count += 1
            if negative_count > 1:
                self.shape = None
                break
        self.dtype = convert_dtype(dtype)
        self._reset()

    def _reset(self):
        self.data = []
        self.lod = [[] for _ in range(self.lod_level)]

    def feed(self, data):
        self._feed_impl_(data, self.lod, self.lod_level)

    def _feed_impl_(self, data, lod, lod_level):
        if lod_level == 0:
            self.data.append(data)
        else:
            lod[0].append(len(data))
            for each_data in data:
                self._feed_impl_(each_data, lod[1:], lod_level - 1)

    def _check_shape(self, shape):
        for s1, s2 in zip(self.shape, shape):
            if s1 != s2 and s1 >= 0 and s2 >= 0:
                raise ValueError(
                    f"Shape not match. What is defined in data layer is {self.shape}, but receive {shape}"
                )

    def done(self):
        arr = np.array(self.data, dtype=self.dtype)
        if self.shape:
            if len(arr.shape) != len(self.shape):
                try:
                    arr = arr.reshape(self.shape)
                except ValueError:
                    raise ValueError(
                        f"Reshape error. What is defined in data layer is {self.shape}, but receive {arr.shape}"
                    )
        t = core.LoDTensor()
        t.set(arr, self.place)
        if self.lod_level > 0:
            t.set_recursive_sequence_lengths(self.lod)
        self._reset()
        return t


class BatchedTensorProvider:
    def __init__(self, feed_list, place, batch_size, generator, drop_last):
        self.place = place
        self.batch_size = batch_size
        self.generator = generator
        self.converters = []
        self.drop_last = drop_last

        for var in feed_list:
            assert var.lod_level == 0, "lod_level must be 0"
            self.converters.append(
                DataToLoDTensorConverter(
                    place=self.place,
                    lod_level=0,
                    shape=var.shape,
                    dtype=var.dtype,
                )
            )

    def _done(self):
        return [c.done() for c in self.converters]

    def __call__(self):
        idx = 0
        for each_sample in self.generator():
            for each_slot, each_converter in zip(each_sample, self.converters):
                each_converter.data.append(each_slot)

            idx += 1
            if idx == self.batch_size:
                idx = 0
                yield self._done()

        if not self.drop_last and idx > 0:
            yield self._done()
        else:
            [c._reset() for c in self.converters]


class DataFeeder:
    """
    :api_attr: Static Graph

    DataFeeder converts the data that returned by a reader into a data
    structure that can feed into Executor. The reader is usually a
    python generator that returns a list of mini-batch data entries.

    Parameters:
        feed_list (list): Variables or names of Variables that need
            to feed.
        place (:ref:`api_paddle_CPUPlace` | :ref:`api_paddle_CUDAPlace` ):
            place indicates the device (CPU | GPU) the data will be fed into, if
            you want to feed data into GPU, please using :code:`base.CUDAPlace(i)`
            (:code:`i` represents the GPU id), or if you want to feed data into CPU,
            please using :code:`base.CPUPlace()`.
        program (:ref:`api_paddle_static_Program` , optional): The Program that will
            feed data into, if program is None, it will use default_main_program().
            Default None.

    Raises:
        :code:`ValueError` - If some Variables are not in this Program.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> from paddle import base

            >>> paddle.enable_static()
            >>> place = paddle.CPUPlace()
            >>> def reader():
            ...     for _ in range(4):
            ...         yield np.random.random([4]).astype('float32'), np.random.random([3]).astype('float32'),
            ...
            >>> main_program = paddle.static.Program()
            >>> startup_program = paddle.static.Program()

            >>> with paddle.static.program_guard(main_program, startup_program):
            ...     data_1 = paddle.static.data(name='data_1', shape=[None, 2, 2], dtype='float32')
            ...     data_2 = paddle.static.data(name='data_2', shape=[None, 1, 3], dtype='float32')
            ...     out = paddle.static.nn.fc(x=[data_1, data_2], size=2)
            ...     # ...
            >>> feeder = base.DataFeeder([data_1, data_2], place)

            >>> exe = paddle.static.Executor(place)
            >>> exe.run(startup_program)

            >>> feed_data = feeder.feed(reader())

            >>> # print feed_data to view feed results
            >>> # print(feed_data['data_1'])
            >>> # print(feed_data['data_2'])

            >>> outs = exe.run(
            ...     program=main_program,
            ...     feed=feed_data,
            ...     fetch_list=[out]
            ... )
            >>> print(outs)

    """

    def __init__(self, feed_list, place, program=None):
        self.feed_dtypes = []
        self.feed_names = []
        self.feed_shapes = []
        self.feed_lod_level = []
        self.place = place
        if in_pir_mode():
            if program is None:
                program = pir.core.default_main_program()
            for each_var in feed_list:
                if isinstance(each_var, str):
                    raise ValueError(
                        "In PIR Mode, Not supported string input yet"
                    )
                if not isinstance(each_var, Value):
                    raise TypeError("Feed list should contain a list of Value")
                self.feed_dtypes.append(each_var.dtype)
                self.feed_names.append(each_var.name)
                self.feed_lod_level.append(0)
                self.feed_shapes.append(each_var.shape)
        else:
            if program is None:
                program = default_main_program()
            for each_var in feed_list:
                if isinstance(each_var, str):
                    each_var = program.block(0).var(each_var)
                if not isinstance(each_var, (Variable, Value)):
                    raise TypeError(
                        "Feed list should contain a list of variable"
                    )
                self.feed_dtypes.append(each_var.dtype)
                self.feed_names.append(each_var.name)
                self.feed_lod_level.append(each_var.lod_level)
                self.feed_shapes.append(each_var.shape)

    def feed(self, iterable):
        """
        According to :code:`feed_list` of :code:`DataFeeder` and :code:`iterable` , converts
        the input into a data structure that can feed into Executor.

        Parameters:
            iterable (generator): user defined python generator to read the raw input data

        Returns:
            :code:`dict`: a :code:`dict` that contains (variable name - converted tensor) pairs

        Example:
            .. code-block:: python

                >>> # In this example, reader - generator will return a list of ndarray of 3 elements
                >>> # feed API will convert each ndarray input into a tensor
                >>> # the return result is a dict with keys: data_1, data_2, data_3
                >>> # result['data_1']  a LoD-Tensor with shape of  [5, 2, 1, 3]. 5 is batch size, and [2, 1, 3] is the real shape of data_1.
                >>> # result['data_2'], result['data_3'] are similar.
                >>> import numpy as np
                >>> import paddle
                >>> from paddle import base

                >>> paddle.enable_static()

                >>> def reader(limit=5):
                ...     for i in range(1, limit + 1):
                ...         yield np.ones([6]).astype('float32') * i , np.ones([1]).astype('int64') * i, np.random.random([9]).astype('float32')
                ...
                >>> data_1 = paddle.static.data(name='data_1', shape=[None, 2, 1, 3])
                >>> data_2 = paddle.static.data(name='data_2', shape=[None, 1], dtype='int64')
                >>> data_3 = paddle.static.data(name='data_3', shape=[None, 3, 3], dtype='float32')
                >>> feeder = base.DataFeeder(['data_1','data_2', 'data_3'], paddle.CPUPlace())

                >>> result = feeder.feed(reader())
                >>> print(result['data_1'])
                >>> print(result['data_2'])
                >>> print(result['data_3'])

        """
        converter = []
        for lod_level, shape, dtype in zip(
            self.feed_lod_level, self.feed_shapes, self.feed_dtypes
        ):
            converter.append(
                DataToLoDTensorConverter(
                    place=self.place,
                    lod_level=lod_level,
                    shape=shape,
                    dtype=dtype,
                )
            )

        def feed_data(converter, data):
            if isinstance(data, (list, tuple)):
                for item in data:
                    feed_data(converter, item)
            else:
                converter.feed(data)

        if paddle.framework.use_pir_api():
            for each_sample in iterable:
                assert len(each_sample) == len(converter), (
                    "The number of fields in data (%d) does not match "
                    + "len(feed_list) (%d)"
                ) % (len(each_sample), len(converter))
                for each_converter, each_slot in zip(converter, each_sample):
                    feed_data(each_converter, each_slot)

        else:
            for each_sample in iterable:
                assert len(each_sample) == len(converter), (
                    "The number of fields in data (%d) does not match "
                    + "len(feed_list) (%d)"
                ) % (len(each_sample), len(converter))
                for each_converter, each_slot in zip(converter, each_sample):
                    each_converter.feed(each_slot)

        ret_dict = {}
        for each_name, each_converter in zip(self.feed_names, converter):
            ret_dict[each_name] = each_converter.done()
        return ret_dict

    def _get_number_of_places_(self, num_places):
        if num_places is not None:
            return int(num_places)
        elif isinstance(self.place, core.CUDAPlace):
            return len(_cuda_ids())
        else:
            return _cpu_num()
