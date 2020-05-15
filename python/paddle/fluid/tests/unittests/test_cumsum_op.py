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
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


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
            'Out': np.flip(
                np.flip(
                    self.inputs['X'], axis=2).cumsum(axis=2), axis=2)
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
            'Out': np.concatenate(
                (np.zeros(
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
            'Out': np.concatenate(
                (np.zeros(
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
            'Out': np.concatenate(
                (np.zeros(
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
            'Out': np.concatenate(
                (np.zeros(
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
            'Out': np.concatenate(
                (np.zeros(
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
            'Out': np.concatenate(
                (np.flip(
                    a[:, :, :-1].cumsum(axis=2), axis=2), np.zeros(
                        (4, 5, 1), dtype=np.float64)),
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
