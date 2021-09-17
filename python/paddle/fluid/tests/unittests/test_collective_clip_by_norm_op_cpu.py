#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestClipByNormOp(OpTest):
    def setUp(self):
        self.dtype = np.float32
        self.shape = (10, 20)
        self.max_relative_error = 0.006
        input = np.random.random((10, 20)).astype(np.float32)
        input[np.abs(input) < self.max_relative_error] = 0.5
        self.op_type = "c_clip_by_norm"
        self.inputs = {'X': [('X', input)]}
        self.attrs = {'max_norm': 0.5, 'ring_id': 0}
        norm = np.sqrt(np.sum(np.square(input)))
        if norm > 0.5:
            output = 0.5 * input / norm
        else:
            output = input
        self.outputs = {'Out': [('Out', output)]}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CPUPlace()
            try:
                self.check_output_with_place(place)
            except NotImplementedError:
                pass


if __name__ == '__main__':
    unittest.main()
