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

import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from op_test import OpTest


class TestGaussianRandomOp(OpTest):
    def setUp(self):
        self.op_type = "gaussian_random"
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": 1.0,
            "std": 2.,
            "seed": 10,
            "use_mkldnn": self.use_mkldnn
        }

        self.outputs = {'Out': np.zeros((123, 92), dtype='float32')}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        self.assertEqual(outs[0].shape, (123, 92))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype("float32")
        hist /= float(outs[0].size)
        data = np.random.normal(size=(123, 92), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype("float32")
        hist2 /= float(outs[0].size)
        self.assertTrue(
            np.allclose(
                hist, hist2, rtol=0, atol=0.01),
            "hist: " + str(hist) + " hist2: " + str(hist2))


# Situation 2: Attr(shape) is a list(with tensor)
class TestGaussianRandomOp_ShapeTensorList(TestGaussianRandomOp):
    def setUp(self):
        '''Test fill_constant op with specified value
        '''
        self.op_type = "gaussian_random"
        self.init_data()
        shape_tensor_list = []
        for index, ele in enumerate(self.shape):
            shape_tensor_list.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.attrs = {
            'shape': self.infer_shape,
            'mean': self.mean,
            'std': self.std,
            'seed': self.seed,
            'use_mkldnn': self.use_mkldnn
        }

        self.inputs = {"ShapeTensorList": shape_tensor_list}
        self.outputs = {'Out': np.zeros((123, 92), dtype='float32')}

    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, 92]
        self.use_mkldnn = False
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10

    def test_check_output(self):
        self.check_output_customized(self.verify_output)


class TestGaussianRandomOp2_ShapeTensorList(
        TestGaussianRandomOp_ShapeTensorList):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, -1]
        self.use_mkldnn = False
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10


class TestGaussianRandomOp3_ShapeTensorList(
        TestGaussianRandomOp_ShapeTensorList):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [123, -1]
        self.use_mkldnn = True
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10


class TestGaussianRandomOp4_ShapeTensorList(
        TestGaussianRandomOp_ShapeTensorList):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [123, -1]
        self.use_mkldnn = False
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10


# Situation 3: shape is a tensor
class TestGaussianRandomOp1_ShapeTensor(TestGaussianRandomOp):
    def setUp(self):
        '''Test fill_constant op with specified value
        '''
        self.op_type = "gaussian_random"
        self.init_data()
        self.use_mkldnn = False

        self.inputs = {"ShapeTensor": np.array(self.shape).astype("int32")}
        self.attrs = {
            'mean': self.mean,
            'std': self.std,
            'seed': self.seed,
            'use_mkldnn': self.use_mkldnn
        }
        self.outputs = {'Out': np.zeros((123, 92), dtype='float32')}

    def init_data(self):
        self.shape = [123, 92]
        self.use_mkldnn = False
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10


if __name__ == "__main__":
    unittest.main()
