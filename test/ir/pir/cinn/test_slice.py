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
from paddle import nn
from paddle.static import InputSpec


class SliceNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        logits = logits[:, 1, :]
        max_out = paddle.max(logits, -1, keepdim=True)
        return logits - max_out


class TestSlice(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.inputs = paddle.randn([1, 256, 3200], dtype="float32")

    def eval(self, use_cinn):
        paddle.seed(2024)
        net = SliceNet()
        input_spec = [
            InputSpec(shape=[None, None, 3200], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.inputs)
        return out

    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        for i in range(len(dy_out)):
            np.testing.assert_allclose(
                cinn_out[i].numpy(), dy_out[i].numpy(), atol=1e-6, rtol=1e-6
            )


if __name__ == '__main__':
    unittest.main()
