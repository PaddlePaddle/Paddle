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

import unittest

from paddle.fluid.tests.unittests.test_gaussian_random_op import TestGaussianRandomOp


class TestMKLDNNGaussianRandomOpSeed10(TestGaussianRandomOp):

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNGaussianRandomOpSeed0(TestGaussianRandomOp):

    def setUp(self):
        TestGaussianRandomOp.setUp(self)
        self.use_mkldnn = True
        self.attrs = {
            "shape": [123, 92],
            "mean": 1.0,
            "std": 2.0,
            "seed": 10,
            "use_mkldnn": self.use_mkldnn
        }


if __name__ == '__main__':
    unittest.main()
