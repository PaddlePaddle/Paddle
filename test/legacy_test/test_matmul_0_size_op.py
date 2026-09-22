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

from op_test import is_custom_device

import paddle
from paddle import _C_ops
from paddle.base import core


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "mamtul 0 size only with in cuda",
)
class TestMatmulDygraph(unittest.TestCase):
    def test_matmul(self):
        x = paddle.ones([0, 128], dtype="float32")
        y = paddle.ones([128, 128], dtype="float32")
        x.stop_gradient = False
        y.stop_gradient = False
        out = paddle.matmul(x, y)

        dz = paddle.ones([0, 128], dtype="float32")

        out = _C_ops.matmul_grad(x, y, dz, False, False)

        self.assertEqual(out[0].shape, x.shape)
        self.assertEqual(out[1].shape, y.shape)

    def test_matmul_grad_zero_k(self):
        # A legit zero-size backward with a matching (K=0) contraction dim
        # must keep working: [M, 0] . [0, N].
        x = paddle.ones([8, 0], dtype="float32")
        y = paddle.ones([0, 4], dtype="float32")
        dz = paddle.ones([8, 4], dtype="float32")
        out = _C_ops.matmul_grad(x, y, dz, False, False)
        self.assertEqual(out[0].shape, x.shape)
        self.assertEqual(out[1].shape, y.shape)

    def test_matmul_grad_k_mismatch(self):
        # The backward matmul_grad must reject a mismatched contraction (K)
        # dim, matching the forward MatmulInferMeta check instead of blindly
        # sharing X/Y meta with dx/dy.
        # No transpose: X[..., K_x], Y[K_y, ...]; K_x != K_y.
        x = paddle.ones([1024, 32], dtype="float32")
        y = paddle.ones([64, 16], dtype="float32")
        dz = paddle.ones([1024, 16], dtype="float32")
        with self.assertRaises(ValueError):
            _C_ops.matmul_grad(x, y, dz, False, False)

        # transpose_x=True: reduction dim of X is the leading axis.
        x = paddle.ones([32, 1024], dtype="float32")
        y = paddle.ones([64, 16], dtype="float32")
        dz = paddle.ones([1024, 16], dtype="float32")
        with self.assertRaises(ValueError):
            _C_ops.matmul_grad(x, y, dz, True, False)

        # transpose_y=True: reduction dim of Y is the trailing axis.
        x = paddle.ones([1024, 32], dtype="float32")
        y = paddle.ones([16, 64], dtype="float32")
        dz = paddle.ones([1024, 16], dtype="float32")
        with self.assertRaises(ValueError):
            _C_ops.matmul_grad(x, y, dz, False, True)


if __name__ == "__main__":
    unittest.main()
