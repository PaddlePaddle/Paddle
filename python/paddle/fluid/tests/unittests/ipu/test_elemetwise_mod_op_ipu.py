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

import unittest

import paddle
import paddle.static
from paddle.fluid.tests.unittests.ipu.test_elemetwise_mul_op_ipu import TestElemetwiseOpBase


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestMod(TestElemetwiseOpBase):
    def set_atol(self):
        self.atol = 1e-7
        self.rtol = 1e-5
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3

    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_mod


if __name__ == "__main__":
    unittest.main()
