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
# model: configs^keypoint^higherhrnet^higherhrnet_hrnet_w32_512_swahr_single_dy2st_train
# method||__add__,method||__radd__,method||__add__,method||__radd__,method||__add__,method||__add__
import unittest

import numpy as np

import paddle


class SIR51(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_269,  # (shape: [], dtype: paddle.float64, stop_gradient: False)
        var_270,  # (shape: [], dtype: paddle.float64, stop_gradient: False)
        var_271,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_272,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_273 = var_269.__add__(var_270)
        var_274 = var_271.__radd__(0)
        var_275 = var_274.__add__(var_272)
        var_276 = var_269.__radd__(0)
        var_277 = var_276.__add__(var_270)
        var_278 = var_277.__add__(var_275)
        return var_273, var_278


class TestSIR51(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1], dtype=paddle.float64),
            paddle.rand(shape=[1], dtype=paddle.float64),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
        )
        self.net = SIR51()

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
