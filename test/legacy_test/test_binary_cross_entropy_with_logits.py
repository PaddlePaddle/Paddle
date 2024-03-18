# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestBinaryCrossEntropyWithLogits(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.x = np.random.randn(10, 5).astype("float32")
        self.y = np.random.randint(0, 2, (10, 5)).astype("float32")
        self.logits = paddle.to_tensor(self.x)
        self.labels = paddle.to_tensor(self.y)
        self.weight = None
        self.reduction = ["none", "mean", "sum"]
        self.pos_weight = None

    def test_binary_cross_entropy_with_logits(self):
        for reduction in self.reduction:
            dynamic_result = (
                paddle.nn.functional.binary_cross_entropy_with_logits(
                    self.logits,
                    self.labels,
                    weight=self.weight,
                    reduction=reduction,
                    pos_weight=self.pos_weight,
                )
            )
            paddle.core._set_prim_all_enabled(True)
            static_result = paddle.jit.to_static(
                paddle.nn.functional.binary_cross_entropy_with_logits,
                full_graph=True,
            )(
                self.logits,
                self.labels,
                weight=self.weight,
                reduction=reduction,
                pos_weight=self.pos_weight,
            )
            paddle.core._set_prim_all_enabled(False)
            if len(dynamic_result.shape) > 0:
                np.testing.assert_allclose(
                    dynamic_result.numpy(), static_result.numpy()
                )
            else:
                np.testing.assert_allclose(
                    dynamic_result.numpy(), static_result.numpy(), rtol=1e-6
                )


class TestBinaryCrossEntropyWithLogits1(TestBinaryCrossEntropyWithLogits):
    def setUp(self):
        super().setUp()
        self.weight = paddle.to_tensor(np.random.randn(10, 5).astype("float32"))


class TestBinaryCrossEntropyWithLogits2(TestBinaryCrossEntropyWithLogits):
    def setUp(self):
        super().setUp()
        self.pos_weight = paddle.to_tensor(np.random.randn(5).astype("float32"))


class TestBinaryCrossEntropyWithLogits3(TestBinaryCrossEntropyWithLogits):
    def setUp(self):
        super().setUp()
        self.weight = paddle.to_tensor(np.random.randn(10, 5).astype("float32"))
        self.pos_weight = paddle.to_tensor(np.random.randn(5).astype("float32"))


if __name__ == "__main__":
    unittest.main()
