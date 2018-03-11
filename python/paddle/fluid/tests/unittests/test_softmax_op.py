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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestSoftmaxOp(OpTest):
    def setUp(self):
        self.op_type = "softmax"
        self.use_cudnn = False
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [10, 10]).astype("float32")
        }
        self.outputs = {
            'Out': np.apply_along_axis(stable_softmax, 1, self.inputs['X'])
        }
        self.attrs = {'use_cudnn': self.use_cudnn, }

    def init_op_type(self):
        pass

    def test_check_output(self):
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)
        else:
            self.check_output()

    def test_check_grad(self):
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ["X"], "Out", max_relative_error=0.01)
        else:
            self.check_grad(["X"], "Out", max_relative_error=0.01)


class TestSoftmaxCUDNNOp(TestSoftmaxOp):
    def init_op_type(self):
        self.use_cudnn = True


if __name__ == "__main__":
    unittest.main()
