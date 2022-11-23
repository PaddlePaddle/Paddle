#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()


class TestMLUCumSumOp(OpTest):

    def setUp(self):
        self.op_type = "cumsum"
        self.set_mlu()
        self.init_dtype()
        self.init_testcase()

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def init_testcase(self):
        self.attrs = {'axis': 2}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=2)}


class TestMLUCumSumOp2(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'axis': -1, 'reverse': True}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {
            'Out': np.flip(np.flip(self.inputs['X'], axis=2).cumsum(axis=2),
                           axis=2)
        }


class TestMLUCumSumOp3(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'axis': 1}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}


class TestMLUCumSumOp4(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'axis': 0}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}


class TestMLUCumSumOp5(TestMLUCumSumOp):

    def init_testcase(self):
        self.inputs = {'X': np.random.random((5, 20)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}


class TestMLUCumSumOp7(TestMLUCumSumOp):

    def init_testcase(self):
        self.inputs = {'X': np.random.random((100)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}


class TestNPUCumSumExclusive1(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 65)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }


class TestNPUCumSumExclusive2(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 888)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (1, 1, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }


class TestNPUCumSumExclusive3(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 888)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }


class TestNPUCumSumExclusive4(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 3049)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (1, 1, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }


class TestNPUCumSumExclusive5(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 3096)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out':
            np.concatenate((np.zeros(
                (4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                           axis=2)
        }


class TestNPUCumSumReverseExclusive(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'axis': 2, 'reverse': True, "exclusive": True}
        a = np.random.random((4, 5, 6)).astype(self.dtype)
        self.inputs = {'X': a}
        a = np.flip(a, axis=2)
        self.outputs = {
            'Out':
            np.concatenate(
                (np.flip(a[:, :, :-1].cumsum(axis=2),
                         axis=2), np.zeros((4, 5, 1), dtype=self.dtype)),
                axis=2)
        }


class TestNPUCumSumWithFlatten1(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'flatten': True}
        self.inputs = {'X': np.random.random((5, 6)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum()}


class TestNPUCumSumWithFlatten2(TestMLUCumSumOp):

    def init_testcase(self):
        self.attrs = {'flatten': True}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum()}


if __name__ == '__main__':
    unittest.main()
