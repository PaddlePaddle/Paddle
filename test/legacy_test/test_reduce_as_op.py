# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
from utils import dygraph_guard

import paddle
from paddle.static import InputSpec

np.random.seed(100)
paddle.seed(100)


def reduce_as_net(x, target):
    return paddle.reduce_as(x, target)


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )


class TestReduceAsOp(OpTest):
    def setUp(self):
        self.init_dtype()
        self.init_shape()
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.x = np.random.random(self.shape_x) + 1j * np.random.random(
                self.shape_y
            )
            self.y = np.random.random(self.shape_x) + 1j * np.random.random(
                self.shape_y
            )
        else:
            self.x = np.random.random(self.shape_x).astype(self.dtype)
            self.y = np.random.random(self.shape_y).astype(self.dtype)
        self.init_attrs()
        self.calc_output()

        self.python_api = paddle.reduce_as
        self.op_type = "reduce_as"
        self.inputs = {'x': self.x, 'target': self.y}
        self.outputs = {'out': self.out}
        self.if_enable_cinn()
        self.prim_op_type = "prim"
        self.public_python_api = paddle.reduce_as

    def init_dtype(self):
        self.dtype = np.float64

    def init_shape(self):
        self.shape_x = [10, 10, 6]
        self.shape_y = [10, 6]

    def init_attrs(self):
        self.attrs = {'dim': [0]}

    def if_enable_cinn(self):
        pass

    def calc_output(self):
        if len(self.attrs['dim']) != 0:
            if 1 in self.shape_y:
                self.out = self.x.sum(
                    axis=tuple(self.attrs['dim']), keepdims=True
                )
            else:
                self.out = self.x.sum(axis=tuple(self.attrs['dim']))
        else:
            self.out = self.x

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['x'], 'out', check_pir=True, check_prim_pir=True)


class TestReduceAsOp2(TestReduceAsOp):
    def init_type(self):
        self.dtype = 'float32'


class TestReduceAsOp3(TestReduceAsOp):
    def init_type(self):
        self.dtype = 'float16'


class TestReduceAsOp4(TestReduceAsOp):
    def init_type(self):
        self.dtype = 'uint16'


class TestReduceAsOp5(TestReduceAsOp):
    def init_type(self):
        self.dtype = 'int16'


class TestReduceAsOp6(TestReduceAsOp):
    def init_type(self):
        self.dtype = 'int64'


class TestReduceAsOp7(TestReduceAsOp):
    def init_type(self):
        self.dtype = 'bool'


class TestReduceAsOp8(TestReduceAsOp):
    def init_type(self):
        self.dtype = 'int32'


class TestReduceAsOp9(TestReduceAsOp):
    def init_type(self):
        self.dtype = 'int8'


class TestReduceAsOp10(TestReduceAsOp):
    def init_type(self):
        self.dtype = 'uint8'


class TestReduceAs_Complex64(TestReduceAsOp):
    def init_type(self):
        self.dtype = np.complex64


class TestReduceAs_Complex128(TestReduceAsOp):
    def init_type(self):
        self.dtype = np.complex128


class TestReduceAsOp13(TestReduceAsOp):
    def init_shape(self):
        self.shape_x = [10, 10, 6]
        self.shape_y = [6]

    def init_attrs(self):
        self.attrs = {'dim': [0, 1]}


class TestReduceAsOp14(TestReduceAsOp):
    def init_shape(self):
        self.shape_x = [10, 10, 6]
        self.shape_y = [10, 10, 6]

    def init_attrs(self):
        self.attrs = {'dim': []}


class TestReduceAsOp15(TestReduceAsOp):
    def init_shape(self):
        self.shape_x = [10, 10, 6, 6]
        self.shape_y = [1, 10, 1, 1]

    def init_attrs(self):
        self.attrs = {'dim': [0, 2, 3]}


class TestReduceAsDynamicShape(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [300, 20, 100]
        self.shape_y = [20, 100]
        self.dtype_x = "float32"
        self.dtype_y = "float32"
        self.init_x_shape = [None, None, 100]
        self.init_y_shape = [None, 100]
        self.x = np.random.random(self.shape_x).astype(self.dtype_x)
        self.y = np.random.random(self.shape_y).astype(self.dtype_y)
        self.net = reduce_as_net
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        if flag == "static":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=self.init_x_shape, dtype=self.dtype_x),
                    InputSpec(shape=self.init_y_shape, dtype=self.dtype_y),
                ],
            )
            fn.eval()
        else:
            fn = self.net
        res = fn(x, y)
        return res

    def test_all_dynamic(self):
        with dygraph_guard():
            res_ref = self.base_net()
            res = self.base_net("static")
            for ref, actual in zip(res_ref, res):
                np.testing.assert_allclose(ref, actual, rtol=self.tol)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
