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

import os
import sys
import unittest
from os.path import dirname

os.environ['FLAGS_prim_forward_blacklist'] = 'pd_op.embedding'
os.environ['FLAGS_enable_auto_recompute'] = '1'

import numpy as np

import paddle

sys.path.append(dirname(dirname(__file__)))
sys.path.append("../")

import llama_test_model


class TestLlamaModel(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.config = llama_test_model.LlamaConfig(num_hidden_layers=2)
        self.input_ids = paddle.to_tensor(
            [
                [
                    1,
                    29871,
                    31201,
                    236,
                    138,
                    141,
                    30287,
                    30557,
                    30015,
                    233,
                    187,
                    172,
                    31969,
                    31325,
                    31043,
                    30374,
                    30024,
                ]
            ],
            dtype="int64",
        )
        self.position_ids = paddle.to_tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
            dtype="int64",
        )
        self.attention_mask = paddle.to_tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype="int64"
        )

    def run_net(self, mode=None):
        paddle.seed(2024)
        net = llama_test_model.LlamaModel(self.config)
        if mode == "cinn":
            net = paddle.jit.to_static(net)

        out = net(self.input_ids, self.position_ids, self.attention_mask)

        loss = out.sum()
        return out.numpy(), np.abs(loss.numpy()).mean()

    def test_static(self):
        dygraph_res = self.run_net()
        cinn_res = self.run_net(mode="cinn")
        for i in range(len(dygraph_res)):
            np.testing.assert_allclose(
                dygraph_res[i],
                cinn_res[i],
                rtol=1e-2,
                atol=1e-2,
                err_msg=f"***** {i}th value check failed ******",
            )


if __name__ == '__main__':
    unittest.main()
