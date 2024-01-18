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
# model: configs^mask_rcnn^mask_rcnn_r101_vd_fpn_1x_coco_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.input.one_hot,method||unsqueeze,api||paddle.tensor.manipulation.expand_as,api||paddle.tensor.search.nonzero,api||paddle.tensor.manipulation.gather_nd,api||paddle.tensor.manipulation.reshape,method||cast,method||unsqueeze,api||paddle.nn.functional.loss.binary_cross_entropy_with_logits
import unittest

import numpy as np

import paddle


class SIR111(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_1661 = self.create_parameter(
            shape=[80, 256, 1, 1],
            dtype=paddle.float32,
        )
        self.var_1662 = self.create_parameter(
            shape=[80],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_1657,  # (shape: [2, 256, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1658,  # (shape: [2], dtype: paddle.int32, stop_gradient: True)
        var_1659,  # (shape: [2, 28, 28], dtype: paddle.int32, stop_gradient: True)
        var_1660,  # (shape: [2], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1663 = paddle.nn.functional.conv._conv_nd(
            var_1657,
            self.var_1661,
            bias=self.var_1662,
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
        var_1664 = paddle.nn.functional.input.one_hot(var_1658, 80)
        var_1665 = var_1664.unsqueeze([2, 3])
        var_1666 = paddle.tensor.manipulation.expand_as(var_1665, var_1663)
        var_1667 = paddle.tensor.search.nonzero(var_1666)
        var_1668 = paddle.tensor.manipulation.gather_nd(var_1663, var_1667)
        var_1669 = paddle.tensor.manipulation.reshape(var_1668, [2, 28, 28])
        var_1670 = var_1659.cast('float32')
        var_1671 = var_1660.unsqueeze([1, 2])
        var_1672 = paddle.nn.functional.loss.binary_cross_entropy_with_logits(
            var_1669, var_1670, weight=var_1671, reduction='mean'
        )
        return var_1672


class TestSIR111(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[2, 256, 28, 28], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
            paddle.randint(
                low=0, high=10, shape=[2, 28, 28], dtype=paddle.int32
            ),
            paddle.rand(shape=[2], dtype=paddle.float32),
        )
        self.net = SIR111()

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
