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

import numpy as np
import utils

import paddle
from paddle.base import core

# GLOG_vmodule=compiler=3 FLAGS_pir_apply_shape_optimization_pass=0 FLAGS_enable_pir_api=1 FLAGS_prim_enable_dynamic=fasle FLAGS_cinn_new_group_scheduler=1 FLAGS_group_schedule_tiling_first=1 FLAGS_cinn_bucket_compile=True FLAGS_support_reduce_stride_read=1 python test_cinn_group_norm_prim.py


class GroupNormSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias):
        return paddle.nn.functional.group_norm(
            x,
            num_groups=32,
            epsilon=1e-6,
            weight=weight,
            bias=bias,
            data_format="NHWC",
        )


class TestGroupNormSubGraph(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn([80, 128, 256, 128], dtype="float16")
        self.x.stop_gradient = False
        self.weight = paddle.randn([128], dtype="float16")
        self.weight.stop_gradient = False
        self.bias = paddle.randn([128], dtype="float16")
        self.bias.stop_gradient = False

    def eval(self, use_prim):
        if self.x.grad is not None:
            self.x.clear_grad()
            self.weight.clear_grad()
            self.bias.clear_grad()

        if use_prim:
            core._set_prim_all_enabled(True)
        net = GroupNormSubGraph()
        # net.eval()
        net = utils.apply_to_static(net, False)

        out = net(self.x, self.weight, self.bias)
        loss = out.sum()
        loss.backward()

        core._set_prim_all_enabled(False)
        return (
            out,
            self.x.gradient(),
            self.weight.gradient(),
            self.bias.gradient(),
        )

    def test_eval(self):
        cinn_out, cinn_x_grad, cinn_weight_grad, cinn_bias_grad = self.eval(
            use_prim=True
        )
        dy_out, dy_x_grad, dy_weight_grad, dy_bias_grad = self.eval(
            use_prim=False
        )
        np.testing.assert_allclose(cinn_x_grad, dy_x_grad, atol=1e-2)
        np.testing.assert_allclose(cinn_weight_grad, dy_weight_grad, atol=1e-6)
        np.testing.assert_allclose(cinn_bias_grad, dy_bias_grad, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
