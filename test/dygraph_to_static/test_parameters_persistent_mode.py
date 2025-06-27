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

import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_phi_only,
    test_pir_only,
    test_sot_mgs0_only,
)

import paddle

# Enable persistent mode for parameters in dy2st
paddle.set_flags({'FLAGS_parameters_persistent_mode_in_dy2st': True})


class NetWithParameters(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = self.create_parameter([in_size, out_size])
        self.bias = self.create_parameter([out_size], is_bias=True)

    def forward(self, x):
        out = paddle.matmul(x, self.weight)
        out = paddle.add(out, self.bias)
        out = paddle.tanh(out)
        return out


class TestParametersPersistentMode(Dy2StTestBase):
    def setUp(self):
        paddle.seed(1127)
        np.random.seed(1127)

    def run_forward(self, net, inputs):
        outs = []
        for data in inputs:
            outs.append(net(data))
        return outs

    @test_pir_only
    def test_persistent_mode(self):
        net = NetWithParameters(10, 3)
        net.eval()
        inputs = [paddle.randn([2, 10], dtype='float32') for _ in range(5)]
        dy_outs = self.run_forward(net, inputs)
        st_net = paddle.jit.to_static(net)
        st_outs = self.run_forward(st_net, inputs)
        for dy_out, st_out in zip(dy_outs, st_outs):
            np.testing.assert_allclose(
                dy_out.numpy(), st_out.numpy(), rtol=1e-05, atol=1e-05
            )

    @test_pir_only
    @test_sot_mgs0_only
    @test_phi_only
    def test_training_mode_error(self):
        net = NetWithParameters(10, 3)
        net.train()
        inputs = [paddle.randn([2, 10], dtype='float32')]
        st_net = paddle.jit.to_static(net)
        with self.assertRaisesRegex(
            RuntimeError,
            "Currently parameters persistent mode only support forward process",
        ):
            self.run_forward(st_net, inputs)


if __name__ == "__main__":
    unittest.main()
