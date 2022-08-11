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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestCumsumOp(unittest.TestCase):

    def run_cases(self):
        data_np = np.arange(12).reshape(3, 4)
        data = paddle.to_tensor(data_np)

        y = paddle.cumsum(data)
        z = np.cumsum(data_np)
        np.testing.assert_array_equal(z, y.numpy())

        y = paddle.cumsum(data, axis=0)
        z = np.cumsum(data_np, axis=0)
        np.testing.assert_array_equal(z, y.numpy())

        y = paddle.cumsum(data, axis=-1)
        z = np.cumsum(data_np, axis=-1)
        np.testing.assert_array_equal(z, y.numpy())

        y = paddle.cumsum(data, dtype='float64')
        self.assertTrue(y.dtype == core.VarDesc.VarType.FP64)

        y = paddle.cumsum(data, dtype=np.int32)
        self.assertTrue(y.dtype == core.VarDesc.VarType.INT32)

        y = paddle.cumsum(data, axis=-2)
        z = np.cumsum(data_np, axis=-2)
        np.testing.assert_array_equal(z, y.numpy())

    def run_static(self, use_gpu=False):
        with fluid.program_guard(fluid.Program()):
            data_np = np.random.random((100, 100)).astype(np.float32)
            x = paddle.static.data('X', [100, 100])
            y = paddle.cumsum(x)
            y2 = paddle.cumsum(x, axis=0)
            y3 = paddle.cumsum(x, axis=-1)
            y4 = paddle.cumsum(x, dtype='float64')
            y5 = paddle.cumsum(x, dtype=np.int32)
            y6 = paddle.cumsum(x, axis=-2)

            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            out = exe.run(feed={'X': data_np},
                          fetch_list=[
                              y.name, y2.name, y3.name, y4.name, y5.name,
                              y6.name
                          ])

            z = np.cumsum(data_np)
            self.assertTrue(np.allclose(z, out[0]))
            z = np.cumsum(data_np, axis=0)
            self.assertTrue(np.allclose(z, out[1]))
            z = np.cumsum(data_np, axis=-1)
            self.assertTrue(np.allclose(z, out[2]))
            self.assertTrue(out[3].dtype == np.float64)
            self.assertTrue(out[4].dtype == np.int32)
            z = np.cumsum(data_np, axis=-2)
            self.assertTrue(np.allclose(z, out[5]))

    def test_cpu(self):
        paddle.disable_static(paddle.fluid.CPUPlace())
        self.run_cases()
        paddle.enable_static()

        self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return
        paddle.disable_static(paddle.fluid.CUDAPlace(0))
        self.run_cases()
        paddle.enable_static()

        self.run_static(use_gpu=True)

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = paddle.static.data('x', [3, 4])
            y = paddle.cumsum(x, name='out')
            self.assertTrue('out' in y.name)


class TestSumOp1(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=2)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp2(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': -1, 'reverse': True}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {
            'Out': np.flip(np.flip(self.inputs['X'], axis=2).cumsum(axis=2),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp3(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 1}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp4(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 0}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp5(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.inputs = {'X': np.random.random((5, 20)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp7(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.inputs = {'X': np.random.random((100)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOpExclusive1(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 65)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (4, 5, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()


class TestSumOpExclusive2(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 888)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (1, 1, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()


class TestSumOpExclusive3(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 888)).astype("float32")
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (4, 5, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()


class TestSumOpExclusive4(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 3049)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (1, 1, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()


class TestSumOpExclusive5(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 3096)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (4, 5, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }

    def test_check_output(self):
        self.check_output()


class TestSumOpReverseExclusive(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, 'reverse': True, "exclusive": True}
        a = np.random.random((4, 5, 6)).astype("float64")
        self.inputs = {'X': a}
        a = np.flip(a, axis=2)
        self.outputs = {
            'Out':
            np.concatenate(
                (np.flip(a[:, :, :-1].cumsum(axis=2),
                         axis=2), np.zeros((4, 5, 1), dtype=np.float64)),
                axis=2)
        }

    def test_check_output(self):
        self.check_output()


class BadInputTest(unittest.TestCase):

    def test_error(self):
        with fluid.program_guard(fluid.Program()):

            def test_bad_x():
                data = [1, 2, 4]
                result = fluid.layers.cumsum(data, axis=0)

            self.assertRaises(TypeError, test_bad_x)


if __name__ == '__main__':
    unittest.main()
