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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core


class TestFillOp(OpTest):
    def setUp(self):
        self.op_type = "fill"
        val = np.random.random(size=[100, 200])
        self.inputs = {}
        self.attrs = {
            'value': val.flatten().tolist(),
            'shape': [100, 200],
            'dtype': int(core.VarDesc.VarType.FP64)
        }
        self.outputs = {'Out': val.astype('float64')}

    def test_check_output(self):
        # TODO(minqiyang): do not support op without kernel
        self.check_output(check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
