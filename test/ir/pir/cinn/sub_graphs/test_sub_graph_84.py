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

# repo: PaddleDetection
# model: configs^ppyoloe^ppyoloe_crn_l_300e_coco_single_dy2st_train
# api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:reshape||api:paddle.nn.functional.activation.softmax||method:transpose||api:paddle.nn.functional.conv._conv_nd||method:squeeze||api:paddle.tensor.manipulation.split||method:__neg__||method:__add__||method:__add__||api:paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[1, 17, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [2541, 2], dtype: paddle.float32, stop_gradient: True)
        var_1,  # (shape: [1, 2541, 68], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2, var_3, var_4 = var_1.shape
        var_5 = var_1.reshape([-1, var_3, 4, 17])
        var_6 = paddle.nn.functional.activation.softmax(var_5)
        var_7 = var_6.transpose([0, 3, 1, 2])
        var_8 = paddle.nn.functional.conv._conv_nd(
            var_7,
            self.parameter_0,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_9 = var_8.squeeze(1)
        var_10, var_11 = paddle.tensor.manipulation.split(var_9, 2, -1)
        var_12 = -var_10
        var_13 = var_12 + var_0
        var_14 = var_11 + var_0
        var_15 = paddle.tensor.manipulation.concat([var_13, var_14], -1)
        return var_15


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[2541, 2], dtype=paddle.float32),
            paddle.rand(shape=[1, 2541, 68], dtype=paddle.float32),
        )
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
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
