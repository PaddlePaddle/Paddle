# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
from os.path import dirname

import numpy as np
from test_infer_sym_shape_utils import (
    TestBase,
    check_infer_results,
)

import paddle
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
from utils import apply_to_static

# NOTE(SigureMo): Disable the CSE optimization to avoid op number change.
paddle.set_flags({"FLAGS_enable_cse_in_dy2st": False})


class MaxMinWithIndexNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min_vals, min_inds = paddle.compat.min(x, dim=-1, keepdim=False)
        max_vals, max_inds = paddle.compat.max(x, dim=-1, keepdim=True)
        return min_vals + max_vals.squeeze(axis=-1), min_inds + max_inds


class MinMaxWithIndexOpInferSymbolicShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(3, 4, 5, 6), np.random.rand(257)]
        self.expected = [
            [
                'shape[S0, S1, S2], data[NULL]',
                'shape[S0, Broadcast(S0, S1), Broadcast(S1, S2), S2], data[NULL]',
            ],
            ['shape[], data[NULL]', 'shape[1], data[NULL]'],
        ]

    def test_eval_symbolic(self):
        net = MaxMinWithIndexNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )
            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(
                net, input_spec, 'builtin.shadow_output', self.expected[i]
            )

        return True


class MinMaxWithIndexRawNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * 2 + 1
        min_vals, min_inds = paddle._C_ops.min_with_index(x, 1, False, True)
        max_vals, max_inds = paddle._C_ops.max_with_index(x, 2, True, True)
        return min_vals + max_vals.squeeze(), min_inds * max_inds


class MinMaxWithIndexOpRawInferShapeTest(TestBase):
    def prepare_data(self):
        self.cases = [np.random.rand(4, 5, 6), np.random.rand(3, 7, 1, 2)]
        self.expected = [
            [
                'shape[], data[NULL]',
                'shape[1, 1, 1], data[NULL]',
            ],
            ['shape[], data[NULL]', 'shape[1, 1, 1, 1], data[NULL]'],
        ]

    @unittest.skipIf(
        not paddle.core.is_compiled_with_cuda(),
        "core is not compiled with CUDA, skipping",
    )
    def test_eval_symbolic(self):
        net = MinMaxWithIndexRawNet()

        for i in range(len(self.cases)):
            x = self.cases[i]
            x_spec = InputSpec(
                shape=[None for index in range(len(x.shape))], dtype='float32'
            )
            input_spec = [x_spec]
            net = apply_to_static(net, False, input_spec)
            net.eval()
            check_infer_results(
                net, input_spec, 'builtin.shadow_output', self.expected[i]
            )

        return True


if __name__ == "__main__":
    unittest.main()
