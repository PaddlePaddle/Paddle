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
# model: configs^rcnn_enhance^faster_rcnn_enhance_3x_coco_single_dy2st_train
# api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.slice,method||__mul__,api||paddle.tensor.manipulation.slice,method||__mul__,api||paddle.tensor.manipulation.slice,method||__mul__,api||paddle.tensor.manipulation.slice,method||__mul__
import unittest

import numpy as np

import paddle


class SIR101(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1183,  # (shape: [8, 4], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1184 = paddle.tensor.manipulation.reshape(
            var_1183, shape=(0, -1, 4)
        )
        var_1185 = paddle.tensor.manipulation.slice(
            var_1184, axes=[2], starts=[0], ends=[1]
        )
        var_1186 = var_1185.__mul__(0.1)
        var_1187 = paddle.tensor.manipulation.slice(
            var_1184, axes=[2], starts=[1], ends=[2]
        )
        var_1188 = var_1187.__mul__(0.1)
        var_1189 = paddle.tensor.manipulation.slice(
            var_1184, axes=[2], starts=[2], ends=[3]
        )
        var_1190 = var_1189.__mul__(0.2)
        var_1191 = paddle.tensor.manipulation.slice(
            var_1184, axes=[2], starts=[3], ends=[4]
        )
        var_1192 = var_1191.__mul__(0.2)
        return var_1190, var_1192, var_1186, var_1188


class TestSIR101(unittest.TestCase):
    def setUp(self):
        self.inputs = (paddle.rand(shape=[8, 4], dtype=paddle.float32),)
        self.net = SIR101()

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
