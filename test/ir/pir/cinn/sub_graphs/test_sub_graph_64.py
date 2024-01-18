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
# model: configs^sparse_rcnn^sparse_rcnn_r50_fpn_3x_pro100_coco_single_dy2st_train
# method||flatten,api||paddle.tensor.ops.sigmoid,method||flatten,api||paddle.tensor.manipulation.concat,method||__gt__,method||all
import unittest

import numpy as np

import paddle


class SIR61(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1232,  # (shape: [1, 100, 80], dtype: paddle.float32, stop_gradient: False)
        var_1233,  # (shape: [1, 100, 4], dtype: paddle.float32, stop_gradient: False)
        var_1238,  # (shape: [2], dtype: paddle.int32, stop_gradient: True)
    ):
        var_1234 = var_1232.flatten(start_axis=0, stop_axis=1)
        var_1235 = paddle.tensor.ops.sigmoid(var_1234)
        var_1236 = var_1233.flatten(start_axis=0, stop_axis=1)
        var_1241 = paddle.tensor.manipulation.concat([var_1238])
        var_1242 = var_1241.__gt__(-1)
        var_1243 = var_1242.all()
        return var_1243, var_1235, var_1241, var_1236


class TestSIR61(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 100, 80], dtype=paddle.float32),
            paddle.rand(shape=[1, 100, 4], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
        )
        self.net = SIR61()

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
