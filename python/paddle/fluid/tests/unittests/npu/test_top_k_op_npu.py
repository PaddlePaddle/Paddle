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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from test_top_k_v2_op_npu import numpy_topk

paddle.enable_static()
SEED = 2021


class TestTopk(OpTest):

    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "top_k"
        self.init_dtype()

        x = np.array([[0.78104149, 0.88745828, 0.32362268],
                      [0.82196718, 0.48763277, 0.42826136],
                      [0.96527182, 0.34851612, 0.12959783]]).astype(self.dtype)

        self.inputs = {'X': x}
        np_out = np.array([[0.88745828], [0.82196718],
                           [0.96527182]]).astype(self.dtype)
        np_indices = np.array([[1], [0], [0]])

        self.attrs = {'k': 1, "axis": -1}
        self.outputs = {'Out': np_out, 'Indices': np_indices}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestTopkV2(OpTest):

    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "top_k"
        self.init_dtype()

        x = np.array([[0.78104149, 0.88745828, 0.32362268],
                      [0.82196718, 0.48763277, 0.42826136],
                      [0.96527182, 0.34851612, 0.12959783]]).astype(self.dtype)

        self.inputs = {'X': x}
        np_out = np.array([[0.88745828, 0.78104149], [0.82196718, 0.48763277],
                           [0.96527182, 0.34851612]]).astype(self.dtype)
        np_indices = np.array([[1, 0], [0, 1], [0, 1]])

        self.attrs = {'k': 2, "axis": -1}
        self.outputs = {'Out': np_out, 'Indices': np_indices}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestTopkV3(OpTest):

    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "top_k"

        self.init_dtype()
        self.set_input_data()
        self.set_attrs()
        output, indices = numpy_topk(self.input_data,
                                     axis=self.axis,
                                     k=self.k,
                                     largest=True)

        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis}
        self.outputs = {'Out': output, 'Indices': indices}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_attrs(self):
        self.k = 3
        self.axis = 1

    def set_input_data(self):
        self.input_data = np.random.choice(10000, size=(10, 20),
                                           replace=False).astype(self.dtype)


if __name__ == '__main__':
    unittest.main()
