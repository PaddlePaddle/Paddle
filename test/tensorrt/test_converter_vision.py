# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from tensorrt_test_base import TensorRTBaseTest

import paddle.nn.functional as F


class TestGridSampleTRTPatternBase(TensorRTBaseTest):
    def setUp(self):
        self.python_api = F.grid_sample
        self.api_args = {
            "x": np.array(
                [[[[-0.6, 0.8, -0.5], [-0.5, 0.2, 1.2], [1.4, 0.3, -0.2]]]]
            ).astype("float32"),
            "grid": np.array(
                [
                    [
                        [[0.2, 0.3], [-0.4, -0.3], [-0.9, 0.3], [-0.9, -0.6]],
                        [[0.4, 0.1], [0.9, -0.8], [0.4, 0.5], [0.5, -0.2]],
                        [[0.1, -0.8], [-0.3, -1.0], [0.7, 0.4], [0.2, 0.8]],
                    ]
                ],
                dtype='float32',
            ),
        }
        self.program_config = {"feed_list": ["x", "grid"]}
        self.min_shape = {"x": [1, 1, 3, 3], "grid": [1, 3, 4, 2]}
        self.opt_shape = {"x": [1, 1, 3, 3], "grid": [1, 3, 4, 2]}
        self.max_shape = {"x": [5, 1, 3, 3], "grid": [5, 3, 4, 2]}


class TestGridSampleTRTPatternCase1(TestGridSampleTRTPatternBase):
    """default:mode='bilinear', padding_mode='zeros', align_corners=True"""

    def test_trt_result(self):
        self.check_trt_result()


class TestGridSampleTRTPatternCase2(TestGridSampleTRTPatternBase):
    """default:mode='nearest', padding_mode='reflection', align_corners=False"""

    def setUp(self):
        super().setUp()
        self.api_args.update(
            {
                "mode": "nearest",
                "padding_mode": "reflection",
                "align_corner": False,
            }
        )

    def test_trt_result(self):
        self.check_trt_result()


class TestGridSampleTRTPatternCase3(TestGridSampleTRTPatternBase):
    """default:mode='nearest', padding_mode='border', align_corners=True"""

    def setUp(self):
        super().setUp()
        self.api_args.update({"mode": "nearest", "padding_mode": "border"})

    def test_trt_result(self):
        self.check_trt_result()


class TestGridSampleTRTPatternCase4(TestGridSampleTRTPatternBase):
    """default:mode='bilinear', padding_mode='border', align_corners=False"""

    def setUp(self):
        super().setUp()
        self.api_args.update(
            {
                "mode": "bilinear",
                "padding_mode": "border",
                "align_corner": False,
            },
        )

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
