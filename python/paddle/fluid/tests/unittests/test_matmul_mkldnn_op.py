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
from test_matmul_op import Generator, generate_compatible_shapes, generate_compatible_batched_shapes


class MKLDNNGenerator(Generator):
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_x(self):
        pass

    def test_check_grad_ignore_y(self):
        pass


# Generate test cases for all possibilities
def inject_test(dim_x, dim_y, trans_x, trans_y):
    test_name = ('TestMatMulOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}'.format(
        dim_x, dim_y, trans_x, trans_y))
    shape_x, shape_y = generate_compatible_batched_shapes(dim_x, dim_y, trans_x,
                                                          trans_y)
    globals()[test_name] = type(test_name, (MKLDNNGenerator, OpTest), {
        'shape_X': shape_x,
        'shape_Y': shape_y,
        'transpose_X': trans_x,
        'transpose_Y': trans_y,
        'use_mkldnn': True,
    })


for dim_X in (1, 2, 3):
    for dim_Y in (1, 2, 3):
        for transpose_x in (False, True):
            for transpose_y in (False, True):
                inject_test(dim_X, dim_Y, transpose_x, transpose_y)

# # Test case n-dim
for dim in [4]:
    for transpose_X in [False, True]:
        for transpose_Y in [False, True]:
            test_name = (
                'TestMatMulOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}'.format(
                    dim, dim, transpose_X, transpose_Y))
            shape_X, shape_Y = generate_compatible_shapes(dim, transpose_X,
                                                          transpose_Y)
            globals()[test_name] = type(test_name, (MKLDNNGenerator, OpTest), {
                'shape_X': shape_X,
                'shape_Y': shape_Y,
                'transpose_X': transpose_X,
                'transpose_Y': transpose_Y,
                'use_mkldnn': True
            })

if __name__ == "__main__":
    unittest.main()
