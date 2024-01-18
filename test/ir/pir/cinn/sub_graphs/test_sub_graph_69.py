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
# model: configs^yolox^yolox_l_300e_coco_single_dy2st_train
# api||paddle.tensor.ops.sigmoid,method||flatten,method||transpose,api||paddle.tensor.manipulation.split,method||flatten,method||transpose,api||paddle.tensor.ops.sigmoid,method||flatten,method||transpose
import unittest

import numpy as np

import paddle


class SIR115(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_507,  # (shape: [1, 5, 36, 36], dtype: paddle.float32, stop_gradient: False)
        var_508,  # (shape: [1, 80, 36, 36], dtype: paddle.float32, stop_gradient: False)
    ):
        var_513 = paddle.tensor.ops.sigmoid(var_508)
        var_514 = var_513.flatten(2)
        var_515 = var_514.transpose([0, 2, 1])
        out = paddle.tensor.manipulation.split(var_507, [4, 1], axis=1)
        var_516 = out[0]
        var_517 = out[1]
        var_518 = var_516.flatten(2)
        var_519 = var_518.transpose([0, 2, 1])
        var_520 = paddle.tensor.ops.sigmoid(var_517)
        var_521 = var_520.flatten(2)
        var_522 = var_521.transpose([0, 2, 1])
        return var_513, var_519, var_517, var_520, var_515, var_522


class TestSIR115(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 5, 36, 36], dtype=paddle.float32),
            paddle.rand(shape=[1, 80, 36, 36], dtype=paddle.float32),
        )
        self.net = SIR115()

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
