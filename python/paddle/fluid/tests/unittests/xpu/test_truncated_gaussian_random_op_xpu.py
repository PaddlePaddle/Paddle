#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append("..")
import unittest
import numpy

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from test_truncated_gaussian_random_op import TestTrunctedGaussianRandomOp

paddle.enable_static()


class TestXPUTrunctedGaussianRandomOp(TestTrunctedGaussianRandomOp):

    def test_xpu(self):
        if paddle.is_compiled_with_xpu():
            self.gaussian_random_test(place=fluid.XPUPlace(0))


if __name__ == "__main__":
    unittest.main()
