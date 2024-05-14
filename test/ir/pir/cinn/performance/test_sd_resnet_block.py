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
import sys
import unittest
from os.path import dirname

sys.path.append(dirname(dirname(__file__)))

import utils

import paddle
from paddle import nn
from paddle.base import core


class ResnetBlock(nn.Layer):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=128, epsilon=1e-06, data_format="NHWC"
        )
        self.conv1 = nn.Conv2D(
            128, 128, kernel_size=3, stride=1, padding=1, data_format="NHWC"
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=128, epsilon=1e-06, data_format="NHWC"
        )
        self.dropout = nn.Dropout(p=0.0, axis=None, mode="upscale_in_train")
        self.conv2 = nn.Conv2D(
            128, 128, kernel_size=[3, 3], padding=1, data_format="NHWC"
        )
        self.nonlinearity = nn.Silu()

    def forward(self, input_tensor):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        output_tensor = input_tensor + hidden_states

        return output_tensor


class TestResnetBlockBase(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.input = paddle.randn([1, 256, 256, 128], dtype="float32")
        self.input.stop_gradient = False

    def eval(self, use_cinn, use_prim=False):
        if use_prim:
            core._set_prim_all_enabled(True)
        net = ResnetBlock()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.input)

        core._set_prim_all_enabled(False)
        return out

    def test_eval(self):
        cinn_outs = self.eval(use_cinn=True, use_prim=True)
        dy_outs = self.eval(use_cinn=False)

        # for cinn_out, dy_out in zip(cinn_outs, dy_outs):
        #     np.testing.assert_allclose(
        #         cinn_out.numpy(), dy_out.numpy(), atol=1e-6
        #     )


# All possible shapes:
# [1, 128, 128, 128]
# [1, 64, 64, 256]
# [1, 64, 64, 512]
# [1, 32, 32, 512]
# [1, 32, 32, 320]
# [1, 16, 16, 320]
# [1, 16, 16, 640]
# [1, 8, 8, 640]
# [1, 8, 8, 1280]
# [1, 4, 4, 1280]
# [1, 4, 4, 2560]
# [1, 32, 32, 960]
# [1, 32, 32, 640]
if __name__ == '__main__':
    unittest.main()
