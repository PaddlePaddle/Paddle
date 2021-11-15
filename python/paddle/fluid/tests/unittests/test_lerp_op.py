# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle


class TestLerp(OpTest):
    def setUp(self):
        self.op_type = "lerp"
        self.init_dtype_type()
        x = np.arange(1., 5.).astype(self.dtype)
        y = np.full(4, 10.).astype(self.dtype)
        w = np.asarray([0.5]).astype(self.dtype)
        self.inputs = {
            'X': x,
            'Y': y,
            'Weight': w,
        }
        self.outputs = {'Out': x + w * (y - x)}

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass
        # self.check_grad(['X', 'Y'], 'Out')


if __name__ == "__main__":
    unittest.main()
