#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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


class TestUnsqueezeOp(OpTest):
    def setUp(self):
        self.op_type = "unsqueeze"
        self.inputs = {'X': np.random.random((3, 4)).astype("float32")}
        self.attrs = {'axes': [0, 2]}
        out = self.inputs['X'][np.newaxis, :, np.newaxis, :]
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", max_relative_error=0.5)


if __name__ == '__main__':
    unittest.main()
