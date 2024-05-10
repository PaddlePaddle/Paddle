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

import numpy as np

os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'

import paddle
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
import utils


class PrepareDecoderAttentionMask(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, cur_len, max_len, unfinish):
        return cur_len < max_len and paddle.any(unfinish)


class TestPrepareDecoderAttentionMask(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.cur_step = paddle.ones([1], dtype="int64")
        self.max_step = paddle.ones([1], dtype="int64")
        self.unfinish = paddle.ones([1, 1], dtype="bool")

    def eval(self, use_cinn=False, mode="static"):
        net = PrepareDecoderAttentionMask()
        input_spec = [
            InputSpec(shape=[1], dtype="int64"),
            InputSpec(shape=[1], dtype="int64"),
            InputSpec(shape=[None, None], dtype="bool"),
        ]
        if mode == "static":
            net = utils.apply_to_static(net, use_cinn, input_spec)
            net.eval()
        out = net(self.cur_step, self.max_step, self.unfinish)
        return out

    def test_eval(self):
        eager_outs = self.eval(mode="eager")
        dy_outs = self.eval(use_cinn=True)

        for cinn_out, dy_out in zip(eager_outs, dy_outs):
            np.testing.assert_allclose(
                cinn_out.numpy(), dy_out.numpy(), atol=1e-8
            )


if __name__ == '__main__':
    unittest.main()
