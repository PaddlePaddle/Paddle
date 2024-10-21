# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import unittest

import numpy as np

import paddle

sys.path.append("..")
from numpy.random import random as rand
from op_test import OpTest, convert_float_to_uint16

import paddle.base.dygraph as dg
from paddle import static
from paddle.base import core

paddle.enable_static()


class TestConjOp(OpTest):
    def setUp(self):
        self.op_type = "conj"
        self.python_api = paddle.tensor.conj
        self.init_dtype_type()
        self.init_input_output()

    def init_dtype_type(self):
        self.dtype = np.complex64

    def init_input_output(self):
        x = (
            np.random.random((12, 14)) + 1j * np.random.random((12, 14))
        ).astype(self.dtype)
        out = np.conj(x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
        )


class TestComplexConjOp(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self._places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_conj_api(self):
        for dtype in self._dtypes:
            input = rand([2, 20, 2, 3]).astype(dtype) + 1j * rand(
                [2, 20, 2, 3]
            ).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = paddle.to_tensor(input)
                    result = paddle.conj(var_x).numpy()
                    target = np.conj(input)
                    np.testing.assert_array_equal(result, target)

    def test_conj_operator(self):
        for dtype in self._dtypes:
            input = rand([2, 20, 2, 3]).astype(dtype) + 1j * rand(
                [2, 20, 2, 3]
            ).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = paddle.to_tensor(input)
                    result = var_x.conj().numpy()
                    target = np.conj(input)
                    np.testing.assert_array_equal(result, target)

    def test_conj_static_mode(self):
        def init_input_output(dtype):
            input = rand([2, 20, 2, 3]).astype(dtype) + 1j * rand(
                [2, 20, 2, 3]
            ).astype(dtype)
            return {'x': input}, np.conj(input)

        for dtype in self._dtypes:
            input_dict, np_res = init_input_output(dtype)
            for place in self._places:
                with static.program_guard(static.Program()):
                    x_dtype = (
                        np.complex64 if dtype == "float32" else np.complex128
                    )
                    x = static.data(
                        name="x", shape=[2, 20, 2, 3], dtype=x_dtype
                    )
                    out = paddle.conj(x)

                    exe = static.Executor(place)
                    out_value = exe.run(feed=input_dict, fetch_list=[out])
                    np.testing.assert_array_equal(np_res, out_value[0])

    def test_conj_api_real_number(self):
        for dtype in self._dtypes:
            input = rand([2, 20, 2, 3]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = paddle.to_tensor(input)
                    result = paddle.conj(var_x).numpy()
                    target = np.conj(input)
                    np.testing.assert_array_equal(result, target)


class Testfp16ConjOp(unittest.TestCase):

    def testfp16(self):
        if paddle.is_compiled_with_cuda():
            input_x = (
                np.random.random((12, 14)) + 1j * np.random.random((12, 14))
            ).astype('float16')
            with static.program_guard(static.Program()):
                x = static.data(name="x", shape=[12, 14], dtype='float16')
                out = paddle.conj(x)
                if paddle.is_compiled_with_cuda():
                    place = paddle.CUDAPlace(0)
                    exe = paddle.static.Executor(place)
                    exe.run(paddle.static.default_startup_program())
                    out = exe.run(feed={'x': input_x}, fetch_list=[out])


class TestConjFP16OP(TestConjOp):
    def init_dtype_type(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestConjBF16(OpTest):
    def setUp(self):
        self.op_type = "conj"
        self.python_api = paddle.tensor.conj
        self.init_dtype_type()
        self.init_input_output()

    def init_dtype_type(self):
        self.dtype = np.uint16

    def init_input_output(self):
        x = (
            np.random.random((12, 14)) + 1j * np.random.random((12, 14))
        ).astype(np.float32)
        out = np.conj(x)

        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_pir=True)


if __name__ == "__main__":
    unittest.main()
