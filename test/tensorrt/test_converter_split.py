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

import paddle


class TestSplitTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.split
        self.api_args = {
            "x": np.random.randn(3, 9, 5).astype(np.float32),
            "num_or_sections": [2, 3, 4],
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 9, 9]}
        self.max_shape = {"x": [10, 9, 10]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
