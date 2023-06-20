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

import unittest

import paddle
from paddle.utils.flops import flops


class TestFLOPSAPI(unittest.TestCase):
    def test_flops(self):
        self.assertTrue(flops('relu', {'X': [[12, 12]]}, {'output': 4}) == 144)
        self.assertTrue(flops('dropout', {}, {'output': 4}) == 0)
        self.assertTrue(
            flops(
                'transpose2',
                {
                    'X': [[12, 12, 12]],
                },
                {},
            )
            == 0
        )
        self.assertTrue(
            flops(
                'reshape2',
                {
                    'X': [[12, 12, 12]],
                },
                {},
            )
            == 0
        )
        self.assertTrue(
            flops(
                'unsqueeze2',
                {
                    'X': [[12, 12, 12]],
                },
                {},
            )
            == 0
        )
        self.assertTrue(
            flops(
                'layer_norm',
                {'Bias': [[128]], 'Scale': [[128]], 'X': [[32, 128, 28, 28]]},
                {'epsilon': 0.01},
            )
            == 32 * 128 * 28 * 28 * 8
        )
        self.assertTrue(
            flops(
                'elementwise_add', {'X': [[12, 12, 12]], 'Y': [[2, 2, 12]]}, {}
            )
            == 12 * 12 * 12
        )
        self.assertTrue(
            flops('gelu', {'X': [[12, 12, 12]]}, {}) == 5 * 12 * 12 * 12
        )
        self.assertTrue(
            flops(
                'matmul',
                {'X': [[3, 12, 12, 8]], 'Y': [[12, 12, 8]]},
                {'transpose_X': False, 'transpose_Y': True},
            )
            == 3 * 12 * 12 * 12 * 2 * 8
        )
        self.assertTrue(
            flops(
                'matmul_v2',
                {'X': [[3, 12, 12, 8]], 'Y': [[12, 12, 8]]},
                {'trans_x': False, 'trans_y': True},
            )
            == 3 * 12 * 12 * 12 * 2 * 8
        )
        self.assertTrue(
            flops('softmax', {'X': [[12, 12, 12]]}, {}) == 3 * 12 * 12 * 12
        )
        self.assertTrue(
            flops('c_embedding', {'Ids': [[12, 12]], 'W': [[12, 12, 3]]}, {})
            == 0
        )
        self.assertTrue(
            flops(
                'elu',
                {
                    'X': [[12, 12]],
                },
                {},
            )
            == 144
        )
        self.assertTrue(
            flops(
                'leaky_relu',
                {
                    'X': [[12, 12]],
                },
                {},
            )
            == 144
        )
        self.assertTrue(
            flops(
                'prelu',
                {
                    'X': [[12, 12]],
                },
                {},
            )
            == 144
        )
        self.assertTrue(
            flops(
                'relu6',
                {
                    'X': [[12, 12]],
                },
                {},
            )
            == 144
        )
        self.assertTrue(
            flops(
                'silu',
                {
                    'X': [[12, 12]],
                },
                {},
            )
            == 144
        )
        self.assertTrue(
            flops(
                'pool',
                {'X': [[12, 12]]},
                {},
            )
            == 12 * 12
        )
        self.assertTrue(
            flops(
                'conv2d',
                {
                    'Bias': [],
                    'Filter': [[3, 3, 2, 2]],
                    'Input': [[8, 3, 4, 4]],
                    'ResidualData': [],
                },
                {
                    'dilations': [1, 1],
                    'groups': 1,
                    'paddings': [1, 1],
                    'strides': [1, 1],
                },
            )
            == 14400
        )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
