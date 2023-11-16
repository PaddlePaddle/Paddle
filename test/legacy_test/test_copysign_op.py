# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

np.random.seed(100)
paddle.seed(100)


def ref_copysign(x, y):
    return np.copysign(x, y)


def ref_grad_copysign(x, y, dout):
    out = np.copysign(x, y)
    return dout * out / x


class TestCopySignOp(OpTest):
    def setUp(self):
        self.op_type = "copysign"
        self.python_api = paddle.copysign
        self.init_config()
        self.inputs = {'x': self.x, 'y': self.y}
        self.target = ref_copysign(self.inputs['x'], self.inputs['y'])
        self.outputs = {'out': self.target}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x'], ['out'])

    def init_config(self):
        self.x = np.random.randn(20, 6).astype('float64')
        self.y = np.random.randn(20, 6).astype('float64')


class TestCopySignAPI(unittest.TestCase):
    def setUp(self):
        self.input_init()
        self.place_init()

    def input_init(self):
        self.x = np.random.randn(20, 6).astype('float64')
        self.y = np.random.randn(20, 6).astype('float64')

    def place_init(self):
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                name='x', shape=self.x.shape, dtype=self.x.dtype
            )
            y = paddle.static.data(
                name='y', shape=self.y.shape, dtype=self.x.dtype
            )
            out = paddle.copysign(x, y)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                paddle.static.default_main_program(),
                feed={"x": self.x, "y": self.y},
                fetch_list=[out],
            )

            out_ref = ref_copysign(self.x, self.y)
            np.testing.assert_allclose(out_ref, res[0])
        paddle.disable_static()

    def test_dygraph_api(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.copysign(x, y)

        out_ref = ref_copysign(self.x, self.y)
        np.testing.assert_allclose(out_ref, out.numpy())
        paddle.enable_static()


class TestCopySignInt32(TestCopySignAPI):
    def input_init(self):
        dtype = np.int32
        self.x = np.zeros(shape=(10, 20)).astype(dtype)
        self.y = np.zeros(shape=(10, 20)).astype(dtype)


class TestCopySignInt64(TestCopySignAPI):
    def input_init(self):
        dtype = np.int64
        self.x = np.zeros(shape=(10, 20)).astype(dtype)
        self.y = np.zeros(shape=(10, 20)).astype(dtype)


class TestCopySignFloat32(TestCopySignAPI):
    def input_init(self):
        dtype = np.float32
        self.x = np.zeros(shape=(10, 20)).astype(dtype)
        self.y = np.zeros(shape=(10, 20)).astype(dtype)


class TestCopySignFloat64(TestCopySignAPI):
    def input_init(self):
        dtype = np.float64
        self.x = np.zeros(shape=(10, 20)).astype(dtype)
        self.y = np.zeros(shape=(10, 20)).astype(dtype)


class TestCopySignZeroCase1(TestCopySignAPI):
    def input_init(self):
        self.x = np.zeros(shape=(10, 20))
        self.y = np.zeros(shape=(10, 20))


class TestCopySignZeroCase2(TestCopySignAPI):
    def input_init(self):
        self.x = np.zeros(shape=(10, 20))
        self.y = np.random.randn(10, 20)


class TestCopySignZeroCase3(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(10, 20)
        self.y = np.zeros(shape=(10, 20))


class TestCopySignZeroDimCase1(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(0, 0)
        self.y = np.random.randn(0, 0)


class TestCopySignZeroDimCase2(TestCopySignAPI):
    def input_init(self):
        self.x = np.random.randn(0, 5, 10)
        self.y = np.random.randn(0, 5, 10)


class TestCopySignSpecialZeroCase1(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([1, 2, 3])
        self.y = np.array([0, +0, -0])


class TestCopySignSpecialZeroCase2(TestCopySignAPI):
    def input_init(self):
        self.x = np.array([0, +0, -0])
        self.y = np.array([1, 2, 3])


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
