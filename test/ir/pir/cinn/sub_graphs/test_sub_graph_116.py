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
# api||paddle.tensor.math.clip,api||paddle.tensor.ops.exp,api||paddle.tensor.ops.exp,method||__rmul__,method||__sub__,method||__rmul__,method||__sub__,method||__rmul__,method||__add__,method||__rmul__,method||__add__,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR103(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1198,  # (shape: [8, 1, 1], dtype: paddle.float32, stop_gradient: False)
        var_1199,  # (shape: [8, 1, 1], dtype: paddle.float32, stop_gradient: False)
        var_1200,  # (shape: [8, 1, 1], dtype: paddle.float32, stop_gradient: False)
        var_1201,  # (shape: [8, 1, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1202 = paddle.tensor.math.clip(
            var_1198, -10000000000.0, 4.135166556742356
        )
        var_1203 = paddle.tensor.ops.exp(var_1201)
        var_1204 = paddle.tensor.ops.exp(var_1202)
        var_1205 = var_1203.__rmul__(0.5)
        var_1206 = var_1199.__sub__(var_1205)
        var_1207 = var_1204.__rmul__(0.5)
        var_1208 = var_1200.__sub__(var_1207)
        var_1209 = var_1203.__rmul__(0.5)
        var_1210 = var_1199.__add__(var_1209)
        var_1211 = var_1204.__rmul__(0.5)
        var_1212 = var_1200.__add__(var_1211)
        var_1213 = paddle.tensor.manipulation.reshape(var_1206, shape=(-1,))
        var_1214 = paddle.tensor.manipulation.reshape(var_1208, shape=(-1,))
        var_1215 = paddle.tensor.manipulation.reshape(var_1210, shape=(-1,))
        var_1216 = paddle.tensor.manipulation.reshape(var_1212, shape=(-1,))
        var_1217 = paddle.tensor.manipulation.concat(
            [var_1213, var_1214, var_1215, var_1216]
        )
        return var_1217


class TestSIR103(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[8, 1, 1], dtype=paddle.float32),
            paddle.rand(shape=[8, 1, 1], dtype=paddle.float32),
            paddle.rand(shape=[8, 1, 1], dtype=paddle.float32),
            paddle.rand(shape=[8, 1, 1], dtype=paddle.float32),
        )
        self.net = SIR103()

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
