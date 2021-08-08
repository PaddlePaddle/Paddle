#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase1(OpTest):
    def setUp(self):
        self.set_npu()
        self.set_example()
        self.op_type = "split"
        self.place = paddle.NPUPlace(0)
        ipt = self.x.astype(self.dtype)
        axis = self.axis if isinstance(self.axis, int) else int(self.axis[0])
        tmp_outs = np.split(
            ipt, axis=axis, indices_or_sections=self.num_or_sections)
        tmp_outs = [o.astype(self.dtype) for o in tmp_outs]
        self.outputs = {'Out': []}
        self.outs = []
        for i, o in enumerate(tmp_outs):
            self.outputs["Out"].append((str(i), o))
            self.outs.append(str(i))

        self.attrs = {"axis": self.axis, "num": self.num_or_sections}
        self.inputs = {}
        self.inputs.update({'X': ipt.astype(self.dtype)})

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], self.outs, check_dygraph=False)

    def set_example(self):
        self.dtype = "float32"
        self.x = np.random.random((2, 4, 6))
        self.axis = 1
        self.num_or_sections = 2


class TestCase2(TestCase1):
    def set_example(self):
        self.dtype = "float32"
        self.x = np.random.random((20, 4, 50))
        self.axis = 0
        self.num_or_sections = 4


# class TestCase3(TestCase1):
#     def set_example(self):
#         self.x=np.random.random((20,4,50)) * 100
#         self.dtype = "int32"
#         self.axis=1
#         self.num_or_sections = 2


class TestCase4(TestCase1):
    def set_example(self):
        self.dtype = "float16"
        self.x = np.random.random((4, 50, 20))
        self.axis = 2
        self.num_or_sections = 4


# Test AxisTensor
# class TestCase4(TestCase1):
#     def set_example(self):
#         self.dtype="float32"
#         self.x = np.random.random((2,20,40))
#         self.num_or_sections = 2
#         self.axis = 2
#     def setUp(self):
#         super().setUp()
#         self.inputs.update({
#             'AxisTensor': np.array([self.axis]).astype("int32")
#         })


# Test Sections
class TestCase5(TestCase1):
    def set_example(self):
        super().set_example()
        self.x = np.random.random((2, 10, 4))
        self.axis = 1
        self.num_or_sections = [2, 4, 8]

    def setUp(self):
        super().setUp()
        self.attrs.update({"sections": [2, 2, 4, 2], "num": 0})


if __name__ == '__main__':
    unittest.main()
