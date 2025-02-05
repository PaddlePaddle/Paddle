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
# api:paddle.nn.functional.conv.conv2d||method:transpose||method:flatten||api:paddle.nn.functional.norm.layer_norm||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:reshape||method:transpose||method:reshape||method:transpose||api:paddle.tensor.linalg.matmul||method:__mul__||api:paddle.nn.functional.activation.softmax||api:paddle.tensor.linalg.matmul||method:transpose||method:reshape||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout||method:__truediv__||method:__add__||api:paddle.nn.functional.norm.layer_norm||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:reshape||method:transpose||method:reshape||method:transpose||api:paddle.tensor.linalg.matmul||method:__mul__||api:paddle.nn.functional.activation.softmax||api:paddle.tensor.linalg.matmul||method:transpose||method:reshape||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout||method:__truediv__||method:__add__||api:paddle.nn.functional.norm.layer_norm||api:paddle.nn.functional.common.linear||method:chunk||api:paddle.nn.functional.activation.gelu||method:__mul__||api:paddle.nn.functional.common.dropout||api:paddle.nn.functional.common.linear||method:__add__||method:reshape||method:transpose||api:paddle.nn.functional.conv.conv2d||method:__add__
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[1280, 1280],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
            shape=[1280, 1280],
            dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
            shape=[1280, 1280],
            dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
            shape=[10240],
            dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
            shape=[1280, 1280, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
            shape=[1280, 1280, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
            shape=[1280, 1280],
            dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
            shape=[5120, 1280],
            dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
            shape=[768, 1280],
            dtype=paddle.float32,
        )
        self.parameter_16 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )
        self.parameter_17 = self.create_parameter(
            shape=[1280, 1280],
            dtype=paddle.float32,
        )
        self.parameter_18 = self.create_parameter(
            shape=[1280, 1280],
            dtype=paddle.float32,
        )
        self.parameter_19 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )
        self.parameter_20 = self.create_parameter(
            shape=[768, 1280],
            dtype=paddle.float32,
        )
        self.parameter_21 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )
        self.parameter_22 = self.create_parameter(
            shape=[1280, 10240],
            dtype=paddle.float32,
        )
        self.parameter_23 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 1280, 1, 1], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [], dtype: paddle.int32, stop_gradient: True)
        var_2,  # (shape: [], dtype: paddle.int32, stop_gradient: True)
        var_3,  # (shape: [1, 1280, 1, 1], dtype: paddle.float32, stop_gradient: False)
        var_4,  # (shape: [1, 4, 768], dtype: paddle.float32, stop_gradient: True)
    ):
        var_5 = paddle.nn.functional.conv.conv2d(
            var_0, self.parameter_10, self.parameter_19, [1, 1], 0, [1, 1], 1
        )
        var_6 = var_5.transpose([0, 2, 3, 1])
        var_7 = var_6.flatten(1, 2)
        var_8 = paddle.nn.functional.norm.layer_norm(
            var_7,
            normalized_shape=[1280],
            weight=self.parameter_1,
            bias=self.parameter_16,
            epsilon=1e-05,
        )
        var_9 = paddle.nn.functional.common.linear(
            x=var_8, weight=self.parameter_5, bias=None, name=None
        )
        var_10 = paddle.nn.functional.common.linear(
            x=var_8, weight=self.parameter_6, bias=None, name=None
        )
        var_11 = paddle.nn.functional.common.linear(
            x=var_8, weight=self.parameter_17, bias=None, name=None
        )
        var_12 = var_9.reshape([0, 0, 8, 160])
        var_13 = var_12.transpose([0, 2, 1, 3])
        var_14 = var_10.reshape([0, 0, 8, 160])
        var_15 = var_14.transpose([0, 2, 1, 3])
        var_16 = var_11.reshape([0, 0, 8, 160])
        var_17 = var_16.transpose([0, 2, 1, 3])
        var_18 = paddle.tensor.linalg.matmul(var_13, var_15, transpose_y=True)
        var_19 = var_18 * 0.07905694150420949
        var_20 = paddle.nn.functional.activation.softmax(var_19, axis=-1)
        var_21 = paddle.tensor.linalg.matmul(var_20, var_17)
        var_22 = var_21.transpose([0, 2, 1, 3])
        var_23 = var_22.reshape([0, 0, 1280])
        var_24 = paddle.nn.functional.common.linear(
            x=var_23, weight=self.parameter_13, bias=self.parameter_3, name=None
        )
        var_25 = paddle.nn.functional.common.dropout(
            var_24,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_26 = var_25 / 1.0
        var_27 = var_26 + var_7
        var_28 = paddle.nn.functional.norm.layer_norm(
            var_27,
            normalized_shape=[1280],
            weight=self.parameter_11,
            bias=self.parameter_21,
            epsilon=1e-05,
        )
        var_29 = paddle.nn.functional.common.linear(
            x=var_28, weight=self.parameter_18, bias=None, name=None
        )
        var_30 = paddle.nn.functional.common.linear(
            x=var_4, weight=self.parameter_15, bias=None, name=None
        )
        var_31 = paddle.nn.functional.common.linear(
            x=var_4, weight=self.parameter_20, bias=None, name=None
        )
        var_32 = var_29.reshape([0, 0, 8, 160])
        var_33 = var_32.transpose([0, 2, 1, 3])
        var_34 = var_30.reshape([0, 0, 8, 160])
        var_35 = var_34.transpose([0, 2, 1, 3])
        var_36 = var_31.reshape([0, 0, 8, 160])
        var_37 = var_36.transpose([0, 2, 1, 3])
        var_38 = paddle.tensor.linalg.matmul(var_33, var_35, transpose_y=True)
        var_39 = var_38 * 0.07905694150420949
        var_40 = paddle.nn.functional.activation.softmax(var_39, axis=-1)
        var_41 = paddle.tensor.linalg.matmul(var_40, var_37)
        var_42 = var_41.transpose([0, 2, 1, 3])
        var_43 = var_42.reshape([0, 0, 1280])
        var_44 = paddle.nn.functional.common.linear(
            x=var_43, weight=self.parameter_0, bias=self.parameter_23, name=None
        )
        var_45 = paddle.nn.functional.common.dropout(
            var_44,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_46 = var_45 / 1.0
        var_47 = var_46 + var_27
        var_48 = paddle.nn.functional.norm.layer_norm(
            var_47,
            normalized_shape=[1280],
            weight=self.parameter_7,
            bias=self.parameter_8,
            epsilon=1e-05,
        )
        var_49 = paddle.nn.functional.common.linear(
            var_48, self.parameter_22, self.parameter_9
        )
        out = var_49.chunk(2, axis=-1)
        var_50 = out[0]
        var_51 = out[1]
        var_52 = paddle.nn.functional.activation.gelu(var_51)
        var_53 = var_50 * var_52
        var_54 = paddle.nn.functional.common.dropout(
            var_53,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_55 = paddle.nn.functional.common.linear(
            var_54, self.parameter_14, self.parameter_2
        )
        var_56 = var_55 + var_47
        var_57 = var_56.reshape([-1, var_1, var_2, 1280])
        var_58 = var_57.transpose([0, 3, 1, 2])
        var_59 = paddle.nn.functional.conv.conv2d(
            var_58, self.parameter_12, self.parameter_4, [1, 1], 0, [1, 1], 1
        )
        var_60 = var_59 + var_3
        return var_60


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 1280, 1, 1], dtype=paddle.float32),
        paddle.randint(low=1, high=2, shape=[1], dtype=paddle.int32),
        paddle.randint(low=1, high=2, shape=[1], dtype=paddle.int32),
        paddle.rand(shape=[1, 1280, 1, 1], dtype=paddle.float32),
        paddle.rand(shape=[1, 4, 768], dtype=paddle.float32),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_paddle_inputs()
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
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
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=False, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
