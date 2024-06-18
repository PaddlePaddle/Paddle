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
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils


class ReduceMean(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, axis=-1):
        out = paddle.mean(x, axis=axis)
        return out


class TestReduceMean(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()
        self.net = ReduceMean()

    def prepare_data(self):
        self.shape = [1, 32, 768]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, x, axis, input_spec, use_cinn):
        net = utils.apply_to_static(self.net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, axis)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_static_axis(self):
        axis = -1
        input_spec = [
            InputSpec(shape=[1, None, 768], dtype='float32'),
        ]
        cinn_out = self.eval(
            x=self.x, axis=axis, input_spec=input_spec, use_cinn=True
        )
        dy_out = self.eval(
            x=self.x, axis=axis, input_spec=input_spec, use_cinn=False
        )
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )

    def test_eval_dynamic_axis(self):
        axis = 1
        input_spec = [
            InputSpec(shape=[1, None, 768], dtype='float32'),
        ]
        cinn_out = self.eval(
            x=self.x, axis=axis, input_spec=input_spec, use_cinn=True
        )
        dy_out = self.eval(
            x=self.x, axis=axis, input_spec=input_spec, use_cinn=False
        )
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )

    def _test_eval_multi_dynamic_axis(self, axis):
        input_spec = [
            InputSpec(shape=[None, None, 768], dtype='float32'),
        ]
        cinn_out = self.eval(
            x=self.x, axis=axis, input_spec=input_spec, use_cinn=True
        )
        dy_out = self.eval(
            x=self.x, axis=axis, input_spec=input_spec, use_cinn=False
        )
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )

    def test_eval_multi_dynamic_axis(self):
        self._test_eval_multi_dynamic_axis(axis=[0])
        self._test_eval_multi_dynamic_axis(axis=[1])
        self._test_eval_multi_dynamic_axis(axis=[0, 1])
        self._test_eval_multi_dynamic_axis(axis=[0, 2])
        self._test_eval_multi_dynamic_axis(axis=[1, 2])
        self._test_eval_multi_dynamic_axis(axis=[0, 1, 2])
        self._test_eval_multi_dynamic_axis(axis=[])


if __name__ == '__main__':
    unittest.main()
