# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# `fused_matmul` is a oneDNN-only op (its kernel is registered only when built
# with oneDNN) and is reachable only through the legacy (non-PIR) static IR.
# This file lives under test/onednn/ so CMake registers it only on WITH_ONEDNN
# builds; non-oneDNN platforms (e.g. Mac-CPU, XPU, DCU) would otherwise hit
# "kernel `fused_matmul` is not registered". OldIrGuard switches to legacy IR
# robustly regardless of the FLAGS_enable_pir_api import order.
import unittest

import paddle
from paddle.pir_utils import OldIrGuard


class TestFusedMatMulOp_BatchBroadcastError(unittest.TestCase):
    """FusedMatmulInferMeta (the `fused_matmul` op) shares the same right-aligned
    batch broadcast loop as MatmulInferMeta, so it must reject non-broadcastable
    batch dims (e.g. 0 vs 10) as well."""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def _append(self, x_shape, y_shape):
        with OldIrGuard():
            main = paddle.static.Program()
            block = main.global_block()
            x = block.create_var(name='fm_x', dtype='float32', shape=x_shape)
            y = block.create_var(name='fm_y', dtype='float32', shape=y_shape)
            out = block.create_var(
                name='fm_out', dtype='float32', shape=[0, 4, 6]
            )
            block.append_op(
                type='fused_matmul',
                inputs={'X': x, 'Y': y},
                outputs={'Out': out},
            )
            return list(out.shape)

    def test_batch_broadcast_error(self):
        with self.assertRaises(ValueError):
            self._append([0, 4, 5], [10, 5, 6])

    def test_broadcastable_allowed(self):
        # 0 vs 1 is broadcastable -> no error.
        self._append([0, 4, 5], [1, 5, 6])

    def test_unknown_vs_zero(self):
        # -1 (unknown) vs 0 -> 0 (the -1 branch of FusedMatmulInferMeta).
        self.assertEqual(self._append([-1, 4, 5], [0, 5, 6]), [0, 4, 6])

    def test_unknown_vs_large(self):
        # -1 vs n (n > 1) -> n.
        self.assertEqual(self._append([-1, 4, 5], [32, 5, 6]), [32, 4, 6])

    def test_unknown_vs_one(self):
        # -1 vs 1 -> -1 (stays unknown).
        self.assertEqual(self._append([-1, 4, 5], [1, 5, 6]), [-1, 4, 6])


if __name__ == '__main__':
    unittest.main()
