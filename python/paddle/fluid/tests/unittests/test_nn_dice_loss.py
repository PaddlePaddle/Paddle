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
import paddle
import paddle.fluid.layers.nn as nn

num_classes = 4
eps = 1e-6


class TestDiceLossValue(unittest.TestCase):

    def test_dice_loss(self):
        input_ = paddle.rand([2, 3, num_classes])
        label_ = paddle.randint(0, num_classes, [2, 3, 1], dtype=paddle.int64)

        input_np, label_np = input_.numpy(), label_.numpy()
        eye_np = np.eye(num_classes)
        label_np = np.float32(eye_np[np.squeeze(label_np)])
        input_np = np.reshape(input_np, [2, -1])
        label_np = np.reshape(label_np, [2, -1])
        intersection_np = np.sum(input_np * label_np, axis=-1)
        union_np = input_np.sum(-1) + label_np.sum(-1)
        dice_np = np.mean(1 - 2 * intersection_np / (union_np + eps))
        dice_paddle = nn.dice_loss(input_, label_, eps)
        np.testing.assert_allclose(dice_np, dice_paddle.numpy(), rtol=1e-05)


class TestDiceLossInvalidInput(unittest.TestCase):

    def test_error(self):

        def test_invalid_dtype():
            input_ = paddle.rand([2, 3, num_classes], dtype=paddle.float32)
            label_ = paddle.randint(0,
                                    num_classes, [2, 3, 1],
                                    dtype=paddle.int64)
            nn.dice_loss(input_, label_.astype(paddle.float32))

        self.assertRaises(AssertionError, test_invalid_dtype)

        def test_zero_shape_input():
            input_ = paddle.rand([0, 3, num_classes], dtype=paddle.float32)
            label_ = paddle.randint(0,
                                    num_classes, [0, 3, 1],
                                    dtype=paddle.int64)
            nn.dice_loss(input_, label_)

        self.assertRaises(AssertionError, test_zero_shape_input)


if __name__ == "__main__":
    unittest.main()
