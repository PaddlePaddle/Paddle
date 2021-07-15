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

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumOp1(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.set_npu()
        self.init_dtype()
        self.init_testcase()

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def init_testcase(self):
        self.attrs = {'axis': 2}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=2)}


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumOp2(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': -1, 'reverse': True}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {
            'Out': np.flip(
                np.flip(
                    self.inputs['X'], axis=2).cumsum(axis=2), axis=2)
        }


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumOp3(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 1}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumOp4(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 0}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumOp5(TestNPUCumSumOp1):
    def init_testcase(self):
        self.inputs = {'X': np.random.random((5, 20)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumOp7(TestNPUCumSumOp1):
    def init_testcase(self):
        self.inputs = {'X': np.random.random((100)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumExclusive1(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 65)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumExclusive2(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 888)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (1, 1, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumExclusive3(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 888)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }


class TestNPUCumSumExclusive4(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 3049)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (1, 1, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumExclusive5(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 3096)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUCumSumReverseExclusive(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, 'reverse': True, "exclusive": True}
        a = np.random.random((4, 5, 6)).astype(self.dtype)
        self.inputs = {'X': a}
        a = np.flip(a, axis=2)
        self.outputs = {
            'Out': np.concatenate(
                (np.flip(
                    a[:, :, :-1].cumsum(axis=2), axis=2), np.zeros(
                        (4, 5, 1), dtype=self.dtype)),
                axis=2)
        }


if __name__ == '__main__':
    unittest.main()
