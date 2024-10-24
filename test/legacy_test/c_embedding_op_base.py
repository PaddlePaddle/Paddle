#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
)

import paddle
from paddle.framework import core

SEED = 2021
np.random.seed(SEED)


def get_c_embedding(start, end, table, ids):
    index = ids.flatten()
    input_mask = (index < start) | (index >= end)
    masked_input = index - start
    masked_input[input_mask] = 0
    output = table[masked_input]
    output[input_mask] = 0.0
    return output


def c_embedding_wrapper(table, index, start_index=0, vocab_size=-1):
    return paddle._C_ops.c_embedding(table, index, start_index, vocab_size)


class TestCEmbeddingCPU(OpTest):
    def setUp(self):
        self.init_dtype()
        self.initcase()
        if core.is_compiled_with_xpu():
            self.__class__.use_xpu = True
        elif core.is_compiled_with_cuda():
            self.__class__.exist_fp64_check_grad = True

    def initcase(self):
        self.op_type = "c_embedding"
        self.python_api = c_embedding_wrapper
        table = np.random.random((17, 64)).astype(self.dtype)
        ids = np.random.randint(low=0, high=17 * 2, size=(2, 4)).astype(
            self.ids_dtype
        )
        self.start_index = 10
        self.end_index = self.start_index + 17
        self.vocab_size = 34

        self.inputs = {'W': table, 'Ids': ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        self.outputs = {'Out': np_out.reshape((2, 4, 64))}
        self.attrs = {
            'start_index': self.start_index,
            'vocab_size': self.vocab_size,
        }
        if core.is_compiled_with_xpu():
            self.__class__.use_xpu = True

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        self.check_grad_with_place(core.CPUPlace(), ['W'], 'Out')

    def init_dtype(self):
        self.dtype = "float32"
        self.ids_dtype = "int64"


class TestCEmbeddingOpBase(TestCEmbeddingCPU):
    def setUp(self):
        self.init_dtype()
        self.initcase()

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))
        elif core.is_compiled_with_xpu():
            self.check_output_with_place(core.XPUPlace(0))
        else:
            current_place = paddle.framework._current_expected_place()
            if isinstance(current_place, paddle.CustomPlace):
                self.check_output_with_place(current_place)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(core.CUDAPlace(0), ['W'], 'Out')
        elif core.is_compiled_with_xpu():
            self.check_grad_with_place(core.XPUPlace(0), ['W'], 'Out')
        else:
            current_place = paddle.framework._current_expected_place()
            if isinstance(current_place, paddle.CustomPlace):
                self.check_grad_with_place(current_place, ['W'], 'Out')

    def init_dtype(self):
        if core.is_compiled_with_cuda():
            self.dtype = "float64"
            self.ids_dtype = "int64"
        elif core.is_compiled_with_xpu():
            self.dtype = "float32"
            self.ids_dtype = "int64"
        else:
            current_place = paddle.framework._current_expected_place()
            if isinstance(current_place, paddle.CustomPlace):
                self.dtype = "float32"
                self.ids_dtype = "int64"


class TestCEmbeddingOpFP32(TestCEmbeddingOpBase):
    def setUp(self):
        self.init_dtype()
        self.initcase()

    def initcase(self):
        self.op_type = "c_embedding"
        self.python_api = c_embedding_wrapper
        table = np.random.random((17, 64)).astype(self.dtype)
        ids = np.random.randint(low=0, high=17 * 2, size=(2, 4)).astype(
            self.ids_dtype
        )
        self.start_index = 10
        ids[0][1] = 12
        ids[0][2] = 12
        ids[1][2] = 12
        ids[1][3] = 12
        self.end_index = self.start_index + 17

        self.inputs = {'W': table, 'Ids': ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        self.outputs = {'Out': np_out.reshape((2, 4, 64))}
        self.attrs = {'start_index': self.start_index}

        if core.is_compiled_with_xpu():
            self.__class__.use_xpu = True
        elif core.is_compiled_with_cuda():
            self.__class__.exist_fp64_check_grad = True

    def init_dtype(self):
        self.dtype = "float32"
        self.ids_dtype = "int32"


class TestCEmbeddingOpFP16(TestCEmbeddingOpBase):
    def setUp(self):
        self.init_dtype()
        self.initcase()

    def initcase(self):
        self.op_type = "c_embedding"
        self.python_api = c_embedding_wrapper
        table = np.random.random((17, 64)).astype(self.dtype)
        ids = np.random.randint(low=0, high=17 * 2, size=(2, 4)).astype(
            self.ids_dtype
        )
        self.start_index = 10
        ids[0][1] = 12
        ids[0][2] = 12
        ids[1][2] = 12
        ids[1][3] = 12
        self.end_index = self.start_index + 17

        self.inputs = {'W': table, 'Ids': ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        self.outputs = {'Out': np_out.reshape((2, 4, 64))}
        self.attrs = {'start_index': self.start_index}

        if core.is_compiled_with_xpu():
            self.__class__.use_xpu = True
        elif core.is_compiled_with_cuda():
            self.__class__.exist_fp64_check_grad = True

    def init_dtype(self):
        self.dtype = "float16"
        self.ids_dtype = "int32"


class TestCEmbeddingOpBF16(TestCEmbeddingOpBase):
    def setUp(self):
        self.init_dtype()
        self.initcase()

    def initcase(self):
        self.op_type = "c_embedding"
        self.python_api = c_embedding_wrapper
        table = np.random.random((17, 64)).astype('float32')
        table_bf16 = convert_float_to_uint16(table)
        table = convert_uint16_to_float(table_bf16)
        ids = np.random.randint(low=0, high=17 * 2, size=(2, 4)).astype(
            self.ids_dtype
        )
        self.start_index = 10
        ids[0][1] = 12
        ids[0][2] = 12
        ids[1][2] = 12
        ids[1][3] = 12
        self.end_index = self.start_index + 17

        self.inputs = {'W': table_bf16, 'Ids': ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        np_out = convert_float_to_uint16(np_out)
        self.outputs = {'Out': np_out.reshape((2, 4, 64))}
        self.attrs = {'start_index': self.start_index}

        if core.is_compiled_with_xpu():
            self.__class__.use_xpu = True
        elif core.is_compiled_with_cuda():
            self.__class__.exist_fp64_check_grad = True

    def init_dtype(self):
        self.dtype = np.uint16
        self.ids_dtype = "int32"


class TestCEmbeddingOpComplex64(TestCEmbeddingOpBase):
    def setUp(self):
        self.init_dtype()
        self.initcase()

    def initcase(self):
        self.op_type = "c_embedding"
        self.python_api = c_embedding_wrapper
        table = (
            np.random.random((17, 64)) + 1j * np.random.random((17, 64))
        ).astype(self.dtype)
        ids = np.random.randint(low=0, high=17 * 2, size=(2, 4)).astype(
            self.ids_dtype
        )
        self.start_index = 10
        ids[0][1] = 12
        ids[0][2] = 12
        ids[1][2] = 12
        ids[1][3] = 12
        self.end_index = self.start_index + 17

        self.inputs = {'W': table, 'Ids': ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        self.outputs = {'Out': np_out.reshape((2, 4, 64))}
        self.attrs = {'start_index': self.start_index}

        if core.is_compiled_with_cuda():
            self.__class__.exist_fp64_check_grad = True

    def init_dtype(self):
        self.dtype = "complex64"
        self.ids_dtype = "int32"


class TestCEmbeddingOpComplex128(TestCEmbeddingOpBase):
    def setUp(self):
        self.init_dtype()
        self.initcase()

    def initcase(self):
        self.op_type = "c_embedding"
        self.python_api = c_embedding_wrapper
        table = (
            np.random.random((17, 64)) + 1j * np.random.random((17, 64))
        ).astype(self.dtype)
        ids = np.random.randint(low=0, high=17 * 2, size=(2, 4)).astype(
            self.ids_dtype
        )
        self.start_index = 10
        ids[0][1] = 12
        ids[0][2] = 12
        ids[1][2] = 12
        ids[1][3] = 12
        self.end_index = self.start_index + 17

        self.inputs = {'W': table, 'Ids': ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        self.outputs = {'Out': np_out.reshape((2, 4, 64))}
        self.attrs = {'start_index': self.start_index}

        if core.is_compiled_with_cuda():
            self.__class__.exist_fp64_check_grad = True

    def init_dtype(self):
        self.dtype = "complex128"
        self.ids_dtype = "int32"


if __name__ == "__main__":
    unittest.main()
