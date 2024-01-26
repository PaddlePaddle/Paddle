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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^Distillation^resnet34_distill_resnet18_afd
# method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||method:pow||method:mean||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||method:reshape||api:paddle.tensor.manipulation.stack
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def process(self, var):
        _var_0 = var.pow(2)
        _var_1 = _var_0.mean(1, keepdim=True)
        _var_2 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            _var_1,
            output_size=(7, 7),
            data_format='NCHW',
            name=None,
        )
        return _var_2.reshape([22, 49])

    def forward(
        self,
        var_0,  # (shape: [22, 64, 56, 56], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [22, 64, 56, 56], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [22, 128, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [22, 128, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_4,  # (shape: [22, 256, 14, 14], dtype: paddle.float32, stop_gradient: False)
        var_5,  # (shape: [22, 256, 14, 14], dtype: paddle.float32, stop_gradient: False)
        var_6,  # (shape: [22, 512, 7, 7], dtype: paddle.float32, stop_gradient: False)
        var_7,  # (shape: [22, 512, 7, 7], dtype: paddle.float32, stop_gradient: False)
    ):
        var_40 = paddle.tensor.manipulation.stack(
            [
                self.process(var_0),
                self.process(var_1),
                self.process(var_2),
                self.process(var_3),
                self.process(var_4),
                self.process(var_5),
                self.process(var_6),
                self.process(var_7),
            ],
            axis=1,
        )
        return var_40


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 64, 56, 56], dtype=paddle.float32),
            paddle.rand(shape=[22, 64, 56, 56], dtype=paddle.float32),
            paddle.rand(shape=[22, 128, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 128, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 256, 14, 14], dtype=paddle.float32),
            paddle.rand(shape=[22, 256, 14, 14], dtype=paddle.float32),
            paddle.rand(shape=[22, 512, 7, 7], dtype=paddle.float32),
            paddle.rand(shape=[22, 512, 7, 7], dtype=paddle.float32),
        )
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    # NOTE prim + cinn lead to error
    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
