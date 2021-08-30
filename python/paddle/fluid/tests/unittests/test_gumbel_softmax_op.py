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
import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
import paddle
paddle.enable_static()


class TestGumbelSoftmaxOp(OpTest):
    def init_attrs(self):
        self.shape = [7, 6]
        self.attrs = {"hard": False, "axis": -1}
        self.count_expected = 7
        self.dtype = "float32"
        self.use_cudnn = False
        self._cpu_only = True

    def verify_output(self, outs):
        out_np = np.array(outs[0])
        out_np.shape = self.shape
        print(out_np.shape)
        print(self.shape)
        self.assertTrue(list(out_np.shape) == self.shape)
        #self.assertEqual(out_np.sum(), self.count_expected)

    def setUp(self):
        self.op_type = "gumbel_softmax"
        self.init_attrs()
        np.random.seed(0)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.zeros(self.shape).astype(self.dtype)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        print("start checkout output customized")
        self.check_output_customized(self.verify_output)

    def test_check_grad(self):
        print("start check_grad")
        self.check_grad(["X"], "Out", max_relative_error=0.01)
