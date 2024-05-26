# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np
from op_test import OpTest

sys.path.append("../deprecated/legacy_test")
from test_gaussian_random_op import TestGaussianRandomOp

import paddle


class TestMKLDNNGaussianRandomOpSeed10(TestGaussianRandomOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.check_pir_onednn = True


class TestMKLDNNGaussianRandomOpSeed0(TestGaussianRandomOp):
    def setUp(self):
        TestGaussianRandomOp.setUp(self)
        self.use_mkldnn = True
        self.check_pir_onednn = True
        self.attrs = {
            "shape": [123, 92],
            "mean": 1.0,
            "std": 2.0,
            "seed": 10,
            "use_mkldnn": self.use_mkldnn,
        }


class TestGaussianRandomOp_ZeroDim(OpTest):
    def setUp(self):
        self.op_type = "gaussian_random"
        self.__class__.op_type = "gaussian_random"
        self.python_api = paddle.normal
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = True
        self.attrs = {
            "shape": [],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "use_mkldnn": self.use_mkldnn,
        }
        paddle.seed(10)

        self.outputs = {'Out': np.random.normal(self.mean, self.std, ())}

    def set_attrs(self):
        self.mean = 1.0
        self.std = 2.0

    # TODO(qun) find a way to check a random scalar
    def test_check_output(self):
        pass

    def test_check_grad(self):
        pass


if __name__ == '__main__':
    unittest.main()
