#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base

class TestBlockDiagError(unittest.TestCase):
    def test_errors(self):
        def test_type_error():
            A = np.array([[1, 2], [3, 4]])
            B = np.array([[5, 6], [7, 8]])
            C = np.array([[9, 10], [11, 12]])
            with paddle.static.program_guard(base.Program()):
                out = paddle.block_diag(A, B ,C)

        self.assertRaises(TypeError, test_type_error)

        def test_dime_error():
            A = paddle.to_tensor([[[1, 2], [3, 4]]])
            B = paddle.to_tensor([[[5, 6], [7, 8]]])
            C = paddle.to_tensor([[[9, 10], [11, 12]]])
            with paddle.static.program_guard(base.Program()):
                out = paddle.block_diag(A, B ,C)
        
        self.assertRaises(ValueError, test_dime_error)


if __name__ == '__main__':
    unittest.main()
