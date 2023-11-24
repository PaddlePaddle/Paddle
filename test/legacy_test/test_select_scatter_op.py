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

import copy
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.framework import core
from paddle.pir_utils import test_with_pir_api

paddle.enable_static()


class TestSelectScatterOp(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "select_scatter"
        self.python_api = paddle.tensor.select_scatter
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.value_np = np.random.random(self.value_shape).astype(
            self.value_type
        )
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(10):
            for j in range(10):
                self.target[i, self.index, j] = self.value_np[i, j]
        self.inputs = {
            'Src': self.xnp,
            'Values': self.value_np,
        }
        self.attrs = {'Axis': self.axis, 'Index': self.index}
        self.outputs = {'Result': self.target}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(["Src", "Values"], "Result", check_pir=True)

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value_shape = (10, 10)
        self.axis = 1
        self.axis_type = "int64"
        self.index = 1
        self.index_type = "int64"


class TestPutAlongAxisFP16Op(TestSelectScatterOp):
    def init_data(self):
        self.dtype = 'float16'
        self.x_type = "float16"
        self.x_shape = (10, 10, 10)
        self.value_type = "float16"
        self.value_shape = (10, 10)
        self.axis = 1
        self.axis_type = "int64"
        self.index = 1
        self.index_type = "int64"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestPutAlongAxisBF16Op(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "select_scatter"
        self.python_api = paddle.tensor.select_scatter
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.value_np = np.random.random(self.value_shape).astype(
            self.value_type
        )
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(10):
            for j in range(10):
                self.target[i, self.index, j] = self.value_np[i, j]
        self.inputs = {
            'Src': self.xnp,
            'Values': self.value_np,
        }
        self.attrs = {'Axis': self.axis, 'Index': self.index}
        self.outputs = {'Result': self.target}

        self.inputs['Src'] = convert_float_to_uint16(self.inputs['Src'])
        self.inputs['Values'] = convert_float_to_uint16(self.inputs['Values'])
        self.outputs['Result'] = convert_float_to_uint16(self.outputs['Result'])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["Src", "Values"], "Result", check_pir=True
        )

    def init_data(self):
        self.dtype = np.uint16
        self.x_type = "float32"
        self.x_shape = (10, 10, 10)
        self.value_type = "float32"
        self.value_shape = (10, 10)
        self.axis = 1
        self.axis_type = "int64"
        self.index = 1
        self.index_type = "int64"


class TestSelectScatterAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 3, 4]
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = [paddle.CPUPlace()]
        self.axis = 1
        self.index = 1
        self.value_shape = [2, 4]
        self.value_np = np.random.random(self.value_shape).astype(np.float32)
        self.x_feed = copy.deepcopy(self.x_np)
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    @test_with_pir_api
    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('Src', self.shape)
                value = paddle.static.data('Values', self.value_shape)
                out = paddle.select_scatter(x, value, self.axis, self.index)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={
                        'Src': self.x_feed,
                        'Values': self.value_np,
                    },
                    fetch_list=[out],
                )

            out_ref = copy.deepcopy(self.x_np)
            for i in range(2):
                for j in range(4):
                    out_ref[i, self.index, j] = self.value_np[i, j]
            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            value_tensor = paddle.to_tensor(self.value_np)
            out = paddle.select_scatter(
                x_tensor, value_tensor, self.axis, self.index
            )
            out_ref = copy.deepcopy(self.x_np)
            for i in range(2):
                for j in range(4):
                    out_ref[i, self.index, j] = self.value_np[i, j]
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_error(self):
        try:
            x_tensor = paddle.to_tensor(self.x_np)
            value_tensor = paddle.to_tensor(self.value_np).astype("int8")
            res = paddle.select_scatter(x_tensor, value_tensor, 1, 1)
        except Exception as error:
            self.assertIsInstance(error, TypeError)

        try:
            x_tensor = paddle.to_tensor(self.x_np)
            value_tensor = paddle.to_tensor(self.value_np).reshape((2, 2, 2))
            res = paddle.select_scatter(x_tensor, value_tensor, 1, 1)
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)

        try:
            x_tensor = paddle.to_tensor(self.x_np)
            value_tensor = paddle.to_tensor([[2, 2], [2, 2]]).astype(np.float32)
            res = paddle.select_scatter(x_tensor, value_tensor, 1, 1)
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
