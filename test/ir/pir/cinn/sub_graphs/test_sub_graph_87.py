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
# model: configs^cascade_rcnn^cascade_rcnn_r50_fpn_1x_coco_single_dy2st_train
# api||paddle.tensor.manipulation.flatten,api||paddle.nn.functional.common.linear,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.common.linear,api||paddle.nn.functional.activation.relu
import unittest

import numpy as np

import paddle


class SIR88(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_1671 = self.create_parameter(
            shape=[12544, 1024],
            dtype=paddle.float32,
        )
        self.var_1676 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.var_1672 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.var_1675 = self.create_parameter(
            shape=[1024, 1024],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_1669,  # (shape: [512, 256, 7, 7], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1670 = paddle.tensor.manipulation.flatten(
            var_1669, start_axis=1, stop_axis=-1
        )
        var_1673 = paddle.nn.functional.common.linear(
            x=var_1670, weight=self.var_1671, bias=self.var_1672, name=None
        )
        var_1674 = paddle.nn.functional.activation.relu(var_1673)
        var_1677 = paddle.nn.functional.common.linear(
            x=var_1674, weight=self.var_1675, bias=self.var_1676, name=None
        )
        var_1678 = paddle.nn.functional.activation.relu(var_1677)
        return var_1678


class TestSIR88(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[512, 256, 7, 7], dtype=paddle.float32),
        )
        self.net = SIR88()

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
