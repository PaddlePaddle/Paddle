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
# model: configs^faster_rcnn^faster_rcnn_swin_tiny_fpn_1x_coco_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,method||flatten,method||transpose,api||paddle.nn.functional.norm.layer_norm,method||transpose,method||reshape,method||flatten,method||transpose,api||paddle.nn.functional.common.dropout
import unittest

import numpy as np

import paddle


class SIR3(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_72 = self.create_parameter(
            shape=[96],
            dtype=paddle.float32,
        )
        self.var_71 = self.create_parameter(
            shape=[96, 3, 4, 4],
            dtype=paddle.float32,
        )
        self.var_77 = self.create_parameter(
            shape=[96],
            dtype=paddle.float32,
        )
        self.var_76 = self.create_parameter(
            shape=[96],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_68,  # (shape: [1, 3, 576, 800], dtype: paddle.float32, stop_gradient: True)
    ):
        var_73 = paddle.nn.functional.conv._conv_nd(
            var_68,
            self.var_71,
            bias=self.var_72,
            stride=[4, 4],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_74 = var_73.flatten(2)
        var_75 = var_74.transpose([0, 2, 1])
        var_78 = paddle.nn.functional.norm.layer_norm(
            var_75,
            normalized_shape=[96],
            weight=self.var_76,
            bias=self.var_77,
            epsilon=1e-05,
        )
        var_79 = var_78.transpose([0, 2, 1])
        var_80 = var_79.reshape([-1, 96, 144, 200])
        var_81 = var_80.flatten(2)
        var_82 = var_81.transpose([0, 2, 1])
        var_83 = paddle.nn.functional.common.dropout(
            var_82,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        return var_83


class TestSIR3(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 3, 576, 800], dtype=paddle.float32),
        )
        self.net = SIR3()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
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
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
