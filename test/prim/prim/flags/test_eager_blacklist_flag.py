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
from paddle.base import core

# core.set_prim_eager_enabled(True)


def fn(primal, cotangent):
    primal = paddle.to_tensor(primal)
    primal.stop_gradient = False
    return paddle.grad(
        paddle.nn.functional.silu(primal), primal, paddle.to_tensor(cotangent)
    )[0]


class TestPrimFlags(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.primal = paddle.to_tensor(
            np.random.rand(100, 100).astype(np.float32)
        )
        self.primal.stop_gradient = False
        self.cotangent = paddle.to_tensor(
            np.random.rand(100, 100).astype(np.float32)
        )

    def test_prim_flags(self):
        origin = fn(self.primal, self.cotangent)
        core.set_prim_eager_enabled(True)
        actual1 = fn(self.primal, self.cotangent)
        np.testing.assert_allclose(origin, actual1, atol=1e-6)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(
                origin,
                actual1,
            )
        core._set_prim_backward_blacklist("silu_grad")
        actual2 = fn(self.primal, self.cotangent)

        np.testing.assert_array_equal(
            origin,
            actual2,
        )


if __name__ == '__main__':
    unittest.main()
