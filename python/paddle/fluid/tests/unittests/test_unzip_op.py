#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from math import log
from math import exp
from op_test import OpTest
import unittest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

class TestUnzipOp(OpTest):
    """
        Test unzip op with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "unzip"
        self.__class__.op_type = "unzip"
        self.__class__.no_need_check_grad = True

        input = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]]
        lod = [0, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]
        self.inputs = {
            'X': np.array(input).astype("float64"),
            'lod': np.array(lod).astype("int32")
        }
        out = [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [100.0, 200.0, 300.0, 400.0], [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        self.outputs = {'Y': np.array(out, dtype=float)}

    def test_check_output(self):
        paddle.enable_static()
        if core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self.check_output(place)


if __name__ == '__main__':
    unittest.main()
