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
# api||paddle.tensor.manipulation.gather,api||paddle.tensor.manipulation.gather,method||cast,api||paddle.nn.functional.loss.binary_cross_entropy_with_logits,api||paddle.tensor.manipulation.gather,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.gather,method||__sub__,api||paddle.tensor.layer_function_generator.abs,method||sum,method||__truediv__,method||__truediv__
import unittest

import numpy as np

import paddle


class SIR54(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_919,  # (shape: [220968], dtype: paddle.float32, stop_gradient: False)
        var_920,  # (shape: [256, 1], dtype: paddle.int64, stop_gradient: True)
        var_921,  # (shape: [220968], dtype: paddle.int32, stop_gradient: True)
        var_922,  # (shape: [5, 1], dtype: paddle.int64, stop_gradient: True)
        var_923,  # (shape: [220968, 4], dtype: paddle.float32, stop_gradient: False)
        var_924,  # (shape: [220968, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_925 = paddle.tensor.manipulation.gather(var_919, var_920)
        var_926 = paddle.tensor.manipulation.gather(var_921, var_920)
        var_927 = var_926.cast('float32')
        var_928 = paddle.nn.functional.loss.binary_cross_entropy_with_logits(
            logit=var_925, label=var_927, reduction='sum'
        )
        var_929 = paddle.tensor.manipulation.gather(var_923, var_922)
        var_930 = paddle.tensor.manipulation.concat([var_924])
        var_931 = paddle.tensor.manipulation.gather(var_930, var_922)
        var_932 = var_929.__sub__(var_931)
        var_933 = paddle.tensor.layer_function_generator.abs(var_932)
        var_934 = var_933.sum()
        var_935 = var_928.__truediv__(256)
        var_936 = var_934.__truediv__(256)
        return var_935, var_936


class TestSIR54(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[220968], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[256, 1], dtype=paddle.int64),
            paddle.randint(low=0, high=10, shape=[220968], dtype=paddle.int32),
            paddle.randint(low=0, high=10, shape=[5, 1], dtype=paddle.int64),
            paddle.rand(shape=[220968, 4], dtype=paddle.float32),
            paddle.rand(shape=[220968, 4], dtype=paddle.float32),
        )
        self.net = SIR54()

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
