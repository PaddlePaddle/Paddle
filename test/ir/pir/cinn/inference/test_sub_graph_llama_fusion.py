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
import sys
import unittest
from os.path import dirname

import numpy as np

import paddle
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        a = paddle.cast(x, "float32")
        b = a + y
        return b.cast("float32")


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 1, 16, 16], dtype=paddle.float32),
            paddle.rand(shape=[1, 32, 16, 16], dtype=paddle.float32),
        )
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                input_spec = [
                    InputSpec(shape=[None, 1, None, None], dtype='float32'),
                    InputSpec(shape=[None, 32, None, None], dtype='float32'),
                ]
                net = utils.apply_to_static(net, True, input_spec)
                # net = paddle.jit.to_static(
                #     net, build_strategy=build_strategy, full_graph=True, input_spec=input_spec
                # )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        # NOTE(Aurelius84): cinn_op.pool2d only support pool_type='avg' under adaptive=True
        paddle.set_flags({"FLAGS_deny_cinn_ops": "pool2d"})
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        # TODO(Aurelius84): It contains reduce operation and atol can't satisfy
        # 1e-8, so we set it to 1e-6.
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
