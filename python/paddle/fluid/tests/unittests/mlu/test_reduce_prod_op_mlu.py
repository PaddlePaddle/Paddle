#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()


def raw_reduce_prod(x, dim=[0], keep_dim=False):
    return paddle.prod(x, dim, keep_dim)


class TestProdOp(OpTest):

    def setUp(self):
        self.op_type = "reduce_prod"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.python_api = raw_reduce_prod
        self.init_data_type()
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.data_type)}
        self.outputs = {'Out': self.inputs['X'].prod(axis=0)}

    def init_data_type(self):
        self.data_type = "float32"

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)


class TestProd6DOp(OpTest):

    def setUp(self):
        self.op_type = "reduce_prod"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.python_api = raw_reduce_prod
        self.init_data_type()
        self.inputs = {
            'X': np.random.random((5, 6, 2, 3, 4, 2)).astype(self.data_type)
        }
        self.attrs = {'dim': [2, 3, 4]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }

    def init_data_type(self):
        self.data_type = "float32"

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)


class TestProd8DOp(OpTest):

    def setUp(self):
        self.op_type = "reduce_prod"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.python_api = raw_reduce_prod
        self.init_data_type()
        self.inputs = {
            'X': np.random.random(
                (2, 5, 3, 2, 2, 3, 4, 2)).astype(self.data_type)
        }
        self.attrs = {'dim': [2, 3, 4]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }

    def init_data_type(self):
        self.data_type = "float32"

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)


if __name__ == '__main__':
    unittest.main()
