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
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest


class TestRandomCropOp(OpTest):
    def setUp(self):
        to_crop = np.random.random((1, 10, 15)).astype("float32")
        self.op_type = "random_crop"
        self.inputs = {'X': to_crop, 'Seed': np.array([10])}
        self.outputs = {'Out': np.array([1, 2, 3]), 'SeedOut': np.array([2])}
        self.attrs = {'shape': [5, 5]}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
