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
# model: ppcls^configs^ImageNet^EfficientNet^EfficientNetB0
# api||paddle.tensor.attribute.shape,method||__getitem__,api||paddle.tensor.random.rand,method||__radd__,api||paddle.tensor.ops.floor,api||paddle.tensor.math.multiply,method||__truediv__,api||paddle.tensor.math.add
import unittest

import numpy as np

import paddle


class SIR130(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_773,  # (shape: [11, 24, 56, 56], dtype: paddle.float32, stop_gradient: False)
        var_774,  # (shape: [11, 24, 56, 56], dtype: paddle.float32, stop_gradient: False)
    ):
        var_775 = paddle.tensor.attribute.shape(var_773)
        var_776 = var_775.__getitem__(0)
        var_777 = paddle.tensor.random.rand(shape=[var_776, 1, 1, 1])
        var_778 = var_777.__radd__(0.975)
        var_779 = paddle.tensor.ops.floor(var_778)
        var_780 = paddle.tensor.math.multiply(var_773, var_779)
        var_781 = var_780.__truediv__(0.975)
        var_782 = paddle.tensor.math.add(var_781, var_774)
        return var_782


class TestSIR130(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[11, 24, 56, 56], dtype=paddle.float32),
            paddle.rand(shape=[11, 24, 56, 56], dtype=paddle.float32),
        )
        self.net = SIR130()

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
