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

import numpy as np

import paddle
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils


class SplitWithNumLayer(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        axis = paddle.full(shape=[1], fill_value=3, dtype='int32')
        axis = paddle.assign(axis)
        num = 4
        return paddle.split(x, num, axis=axis)


class TestSplitWithNum(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [4, 5, 6, 8]
        self.x = paddle.randn(self.shape, dtype='float16')
        self.x.stop_gradient = True

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 0)
        # utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: })

    def eval(self, use_cinn):
        net = SplitWithNumLayer()
        input_spec = [InputSpec([None, None, None, 8], dtype='float16')]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        outs = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return outs

    def test_eval(self):
        cinn_outs = self.eval(use_cinn=True)
        dygraph_outs = self.eval(use_cinn=False)
        assert len(cinn_outs) == len(dygraph_outs)
        for i in range(len(cinn_outs)):
            np.testing.assert_allclose(
                cinn_outs[i].numpy(),
                dygraph_outs[i].numpy(),
                atol=1e-6,
                rtol=1e-6,
            )


if __name__ == '__main__':
    unittest.main()
