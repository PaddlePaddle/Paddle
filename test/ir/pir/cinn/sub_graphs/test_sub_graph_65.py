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
# api||paddle.tensor.manipulation.concat,method||__pow__,method||__rmul__,method||__rsub__,method||__add__,method||log,method||__neg__,method||__mul__,method||__rsub__,method||__pow__,method||__rmul__,method||__add__,method||log,method||__neg__,method||__mul__,api||paddle.tensor.manipulation.gather,api||paddle.tensor.manipulation.gather,method||__sub__,method||unsqueeze,api||paddle.tensor.manipulation.concat,method||unsqueeze,method||tile,method||flatten,api||paddle.tensor.manipulation.concat,method||__truediv__,method||__truediv__,method||unsqueeze,api||paddle.nn.functional.loss.l1_loss,method||sum
import unittest

import numpy as np

import paddle


class SIR62(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1244,  # (shape: [100, 80], dtype: paddle.float32, stop_gradient: False)
        var_1245,  # (shape: [2], dtype: paddle.int32, stop_gradient: True)
        var_1246,  # (shape: [100, 4], dtype: paddle.float32, stop_gradient: False)
        var_1247,  # (shape: [2, 4], dtype: paddle.float32, stop_gradient: True)
        var_1249,  # (shape: [4], dtype: paddle.float32, stop_gradient: True)
        var_1250,  # (shape: [2, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1251 = paddle.tensor.manipulation.concat([var_1247])
        var_1252 = var_1244.__pow__(2.0)
        var_1253 = var_1252.__rmul__(0.75)
        var_1254 = var_1244.__rsub__(1)
        var_1255 = var_1254.__add__(1e-08)
        var_1256 = var_1255.log()
        var_1257 = var_1256.__neg__()
        var_1258 = var_1253.__mul__(var_1257)
        var_1259 = var_1244.__rsub__(1)
        var_1260 = var_1259.__pow__(2.0)
        var_1261 = var_1260.__rmul__(0.25)
        var_1262 = var_1244.__add__(1e-08)
        var_1263 = var_1262.log()
        var_1264 = var_1263.__neg__()
        var_1265 = var_1261.__mul__(var_1264)
        var_1266 = paddle.tensor.manipulation.gather(var_1265, var_1245, axis=1)
        var_1267 = paddle.tensor.manipulation.gather(var_1258, var_1245, axis=1)
        var_1268 = var_1266.__sub__(var_1267)
        var_1269 = var_1249.unsqueeze(0)
        var_1270 = paddle.tensor.manipulation.concat([var_1269])
        var_1271 = var_1270.unsqueeze(1)
        var_1272 = var_1271.tile([1, 100, 1])
        var_1273 = var_1272.flatten(start_axis=0, stop_axis=1)
        var_1274 = paddle.tensor.manipulation.concat([var_1250])
        var_1275 = var_1246.__truediv__(var_1273)
        var_1276 = var_1251.__truediv__(var_1274)
        var_1277 = var_1275.unsqueeze(-2)
        var_1278 = paddle.nn.functional.loss.l1_loss(
            var_1277, var_1276, reduction='none'
        )
        var_1279 = var_1278.sum(-1)
        return var_1251, var_1279, var_1268


class TestSIR62(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[100, 80], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
            paddle.rand(shape=[100, 4], dtype=paddle.float32),
            paddle.rand(shape=[2, 4], dtype=paddle.float32),
            paddle.rand(shape=[4], dtype=paddle.float32),
            paddle.rand(shape=[2, 4], dtype=paddle.float32),
        )
        self.net = SIR62()

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
