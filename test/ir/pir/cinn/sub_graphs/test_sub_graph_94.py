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
# api||paddle.tensor.manipulation.gather,method||__eq__,api||paddle.tensor.creation.ones_like,method||__mul__,api||paddle.tensor.search.where,method||__eq__,api||paddle.tensor.creation.ones_like,method||__mul__,api||paddle.tensor.search.where
import unittest

import numpy as np

import paddle


class SIR64(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1243,  # (shape: [2002], dtype: paddle.int64, stop_gradient: True)
        var_1244,  # (shape: [2002], dtype: paddle.int32, stop_gradient: True)
        var_1245,  # (shape: [2], dtype: paddle.int32, stop_gradient: True)
    ):
        var_1246 = paddle.tensor.manipulation.gather(var_1245, var_1243)
        var_1247 = var_1244.__eq__(0)
        var_1248 = paddle.tensor.creation.ones_like(var_1246)
        var_1249 = var_1248.__mul__(80)
        var_1250 = paddle.tensor.search.where(var_1247, var_1249, var_1246)
        var_1251 = var_1244.__eq__(-1)
        var_1252 = paddle.tensor.creation.ones_like(var_1250)
        var_1253 = var_1252.__mul__(-1)
        var_1254 = paddle.tensor.search.where(var_1251, var_1253, var_1250)
        return var_1254


class TestSIR64(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.randint(low=0, high=10, shape=[2002], dtype=paddle.int64),
            paddle.randint(low=0, high=10, shape=[2002], dtype=paddle.int32),
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
        )
        self.net = SIR64()

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
