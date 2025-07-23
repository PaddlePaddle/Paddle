# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_places

import paddle

num_classes = 4
eps = 1e-6


class TestDiceLossOpApi_ZeroSize(unittest.TestCase):
    def test_api_with_dygraph(self):
        for place in get_places():
            paddle.disable_static(place)
            input = paddle.randn([0, 2]).astype(paddle.float64)
            input.stop_gradient = False
            label = paddle.randn([0, 1]).astype(paddle.int64)
            label.stop_gradient = False
            out = paddle.nn.functional.dice_loss(input, label, 1e-5)
            np.testing.assert_allclose(out.numpy(), paddle.nan)
            out.sum().backward()
            np.testing.assert_allclose(input.grad.shape, input.shape)


if __name__ == "__main__":
    unittest.main()
