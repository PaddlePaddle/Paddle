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
# method||clone,method||unbind,method||__rmul__,method||__sub__,method||__rmul__,method||__sub__,method||__rmul__,method||__add__,method||__rmul__,method||__add__,api||paddle.tensor.manipulation.stack,method||unsqueeze,method||unsqueeze,method||__mul__,method||unsqueeze,method||tile,method||clone
import unittest

import numpy as np

import paddle


class SIR34(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_258 = self.create_parameter(
            shape=[100, 4],
            dtype=paddle.float32,
        )
        self.var_276 = self.create_parameter(
            shape=[100, 256],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_257,  # (shape: [1, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_259 = self.var_258.clone()
        out = var_259.unbind(-1)
        var_260 = out[0]
        var_261 = out[1]
        var_262 = out[2]
        var_263 = out[3]
        var_264 = var_262.__rmul__(0.5)
        var_265 = var_260.__sub__(var_264)
        var_266 = var_263.__rmul__(0.5)
        var_267 = var_261.__sub__(var_266)
        var_268 = var_262.__rmul__(0.5)
        var_269 = var_260.__add__(var_268)
        var_270 = var_263.__rmul__(0.5)
        var_271 = var_261.__add__(var_270)
        var_272 = paddle.tensor.manipulation.stack(
            [var_265, var_267, var_269, var_271], axis=-1
        )
        var_273 = var_272.unsqueeze(0)
        var_274 = var_257.unsqueeze(-2)
        var_275 = var_273.__mul__(var_274)
        var_277 = self.var_276.unsqueeze(0)
        var_278 = var_277.tile([1, 1, 1])
        var_279 = var_278.clone()
        return var_279, var_275


class TestSIR34(unittest.TestCase):
    def setUp(self):
        self.inputs = (paddle.rand(shape=[1, 4], dtype=paddle.float32),)
        self.net = SIR34()

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
