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

# The `fused_matmul` op is only reachable through the legacy (non-PIR) static
# IR, so disable PIR before importing paddle.
import os

os.environ['FLAGS_enable_pir_api'] = '0'

import unittest

import paddle


class TestFusedMatMulOp_BatchBroadcastError(unittest.TestCase):
    """FusedMatmulInferMeta (the `fused_matmul` op) shares the same equal-rank
    batch loop as MatmulInferMeta, so it must reject non-broadcastable batch
    dims (e.g. 0 vs 10) as well."""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def _append(self, x_shape, y_shape):
        main = paddle.static.Program()
        block = main.global_block()
        x = block.create_var(name='fm_x', dtype='float32', shape=x_shape)
        y = block.create_var(name='fm_y', dtype='float32', shape=y_shape)
        out = block.create_var(name='fm_out', dtype='float32', shape=[0, 4, 6])
        block.append_op(
            type='fused_matmul',
            inputs={'X': x, 'Y': y},
            outputs={'Out': out},
        )

    def test_batch_broadcast_error(self):
        with self.assertRaises(ValueError):
            self._append([0, 4, 5], [10, 5, 6])

    def test_broadcastable_allowed(self):
        # 0 vs 1 is broadcastable -> no error.
        self._append([0, 4, 5], [1, 5, 6])


if __name__ == '__main__':
    unittest.main()
