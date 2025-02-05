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

# repo: diffusers_sub_graph
# model: stable_diffusion
# api:paddle.nn.functional.activation.silu||api:paddle.nn.functional.common.dropout||api:paddle.nn.functional.conv.conv2d||api:paddle.nn.functional.conv.conv2d||method:__add__||method:__truediv__
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.parameter_1 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Constant(2.0),
        )
        self.parameter_2 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Constant(3.0),
        )
        self.parameter_3 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Constant(4.0),
        )

    def forward(
        self,
        var_0,  # (shape: [1, 256, 4, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        return paddle.nn.functional.batch_norm(
            var_0,
            self.parameter_0,
            self.parameter_1,
            self.parameter_2,
            self.parameter_3,
            training=True,
            use_global_stats=False,
        )


def create_paddle_inputs():
    inputs = (paddle.rand(shape=[16, 32, 12, 12], dtype=paddle.float32),)
    inputs[0].stop_gradient = False
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_paddle_inputs()
        self.net = LayerCase()

    def train(self, to_static, with_prim=False, with_cinn=False):
        paddle.seed(123)
        net = LayerCase()
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)

        outs = net(*self.inputs)
        loss = outs[0]
        loss.backward()
        return (
            outs,
            net.state_dict()["parameter_0"],
            net.state_dict()["parameter_1"],
            net.state_dict()["parameter_2"],
            net.state_dict()["parameter_3"],
        )

    def test_ast_prim_cinn(self):
        st_out, st_p0, st_p1, st_p2, st_p3 = self.train(to_static=True)
        cinn_out, cinn_p0, cinn_p1, cinn_p2, cinn_p3 = self.train(
            to_static=True, with_prim=True, with_cinn=True
        )

        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(
                st.numpy(), cinn.numpy(), atol=1e-5, rtol=1e-4
            )

        for st, cinn in zip(
            paddle.utils.flatten(st_p0), paddle.utils.flatten(cinn_p0)
        ):
            np.testing.assert_allclose(
                st.numpy(), cinn.numpy(), atol=1e-5, rtol=1e-4
            )

        for st, cinn in zip(
            paddle.utils.flatten(st_p1), paddle.utils.flatten(cinn_p1)
        ):
            np.testing.assert_allclose(
                st.numpy(), cinn.numpy(), atol=1e-5, rtol=1e-4
            )

        for st, cinn in zip(
            paddle.utils.flatten(st_p2), paddle.utils.flatten(cinn_p2)
        ):
            np.testing.assert_allclose(
                st.numpy(), cinn.numpy(), atol=1e-5, rtol=1e-4
            )

        for st, cinn in zip(
            paddle.utils.flatten(st_p3), paddle.utils.flatten(cinn_p3)
        ):
            np.testing.assert_allclose(
                st.numpy(), cinn.numpy(), atol=1e-5, rtol=1e-4
            )


if __name__ == '__main__':
    unittest.main()
