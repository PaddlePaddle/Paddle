# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from os.path import dirname

import numpy as np

sys.path.append(dirname(dirname(__file__)))

import unittest

import utils

import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec


def reduce_sum(x):
    return paddle.sum(x, axis=-1)


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = F.one_hot

    def forward(self, x, num_classes):
        out = self.fn(x, num_classes)
        return out


class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        x_shape = [10, 128]
        tensor_x = np.random.random(x_shape).astype("int64")
        self.num_classes = 10
        self.x = paddle.to_tensor(tensor_x)
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet()
        input_spec = [
            InputSpec(shape=[None, 128], dtype='int64'),
            # InputSpec(shape=[10], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.num_classes)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)

        return out

    def test_eval_symbolic(self):
        dy_out = self.eval_symbolic(use_cinn=False)
        print(dy_out.numpy())
        cinn_out = self.eval_symbolic(use_cinn=True)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-4)


if __name__ == '__main__':
    unittest.main()
