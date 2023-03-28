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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import fluid
from paddle.fluid import Program, core, program_guard


class TestInstanceNorm(unittest.TestCase):
    def test_error(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            places.append(fluid.CUDAPlace(0))
        for p in places:

            def error1d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                instance_norm1d = paddle.nn.InstanceNorm1D(1)
                instance_norm1d(fluid.dygraph.to_variable(x_data_4))

            def error2d():
                x_data_3 = np.random.random(size=(2, 1, 3)).astype('float32')
                instance_norm2d = paddle.nn.InstanceNorm2D(1)
                instance_norm2d(fluid.dygraph.to_variable(x_data_3))

            def error3d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                instance_norm3d = paddle.nn.InstanceNorm3D(1)
                instance_norm3d(fluid.dygraph.to_variable(x_data_4))

            def weight_bias_false():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                instance_norm3d = paddle.nn.InstanceNorm3D(
                    1, weight_attr=False, bias_attr=False
                )

            with fluid.dygraph.guard(p):
                weight_bias_false()
                self.assertRaises(ValueError, error1d)
                self.assertRaises(ValueError, error2d)
                self.assertRaises(ValueError, error3d)

    def test_dygraph(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.InstanceNorm2D(shape[1])
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def compute_v2(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.InstanceNorm2D(shape[1])
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)

    def test_static(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            exe = fluid.Executor(p)
            shape = [4, 10, 16, 16]

            def compute_v1(x_np):
                with program_guard(Program(), Program()):
                    ins = paddle.nn.InstanceNorm2D(shape[1])
                    x = paddle.static.data(
                        name='x', shape=x_np.shape, dtype=x_np.dtype
                    )
                    y = ins(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    ins = paddle.nn.InstanceNorm2D(shape[1])
                    x = paddle.static.data(
                        name='x', shape=x_np.shape, dtype=x_np.dtype
                    )
                    y = ins(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)


class TestInstanceNormFP32OP(OpTest):
    def setUp(self):
        '''Test instance_norm op with default value'''
        self.op_type = "instance_norm"
        self.python_api = paddle.nn.functional.instance_norm
        self.eps = 1e-5
        self.init_dtype()
        self.init_shape()
        self.init_value()
        self.atol = 1e-3
        self.max_relative_error = 5e-3
        self.inputs = {'X': self.value}
        self.attrs = {
            'epsilon': self.eps,
            'momentum': 0.9,
            'data_format': "NCHW",
        }
        self.outputs = {'Out': self.compute_output(self.value)}

    def compute_output(self, x):
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        gamma = np.ones((1, x.shape[1], 1, 1))
        beta = np.zeros((1, x.shape[1], 1, 1))
        x_norm = gamma * x_norm + beta
        return x_norm

    def test_check_output(self):
        self.check_output(check_eager=True, atol=self.atol)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=self.max_relative_error
        )

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [4, 10, 4, 4]

    def init_value(self):
        self.value = np.random.random(self.shape).asytpe(self.dtype)


class TestInstanceNormFP16OP(TestInstanceNormFP32OP):
    def init_dtype(self):
        self.dtype = np.float16


class TestInstanceNormBF16OP(TestInstanceNormFP32OP):
    def setUp(self):
        self.atol = 1e-2
        self.max_relative_error = 1e-2
        self.inputs = {'X': convert_float_to_uint16(self.value)}
        self.outputs = {
            'Out': convert_float_to_uint16(self.compute_output(self.value))
        }


if __name__ == '__main__':
    unittest.main()
