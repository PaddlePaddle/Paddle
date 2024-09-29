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
# api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.gather||method:cast||api:paddle.nn.functional.loss.binary_cross_entropy_with_logits||api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.gather||method:__sub__||api:paddle.tensor.abs||method:sum||method:__truediv__||method:__truediv__
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [171888], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [256, 1], dtype: paddle.int64, stop_gradient: True)
        var_2,  # (shape: [171888], dtype: paddle.int32, stop_gradient: True)
        var_3,  # (shape: [7, 1], dtype: paddle.int64, stop_gradient: True)
        var_4,  # (shape: [171888, 4], dtype: paddle.float32, stop_gradient: False)
        var_5,  # (shape: [171888, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_6 = paddle.tensor.manipulation.gather(var_0, var_1)
        var_7 = paddle.tensor.manipulation.gather(var_2, var_1)
        var_8 = var_7.cast('float32')
        var_9 = paddle.nn.functional.loss.binary_cross_entropy_with_logits(
            logit=var_6, label=var_8, reduction='sum'
        )
        var_10 = paddle.tensor.manipulation.gather(var_4, var_3)
        var_11 = paddle.tensor.manipulation.concat([var_5])
        var_12 = paddle.tensor.manipulation.gather(var_11, var_3)
        var_13 = var_10 - var_12
        var_14 = paddle.tensor.abs(var_13)
        var_15 = var_14.sum()
        var_16 = var_9 / 256
        var_17 = var_15 / 256
        return var_16, var_17


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, 1),
                dtype=paddle.int64,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.int32, name=None, stop_gradient=True
            ),
            InputSpec(
                shape=(-1, 1),
                dtype=paddle.int64,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[171888], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[256, 1], dtype=paddle.int64),
            paddle.randint(low=0, high=10, shape=[171888], dtype=paddle.int32),
            paddle.randint(low=0, high=10, shape=[7, 1], dtype=paddle.int64),
            paddle.rand(shape=[171888, 4], dtype=paddle.float32),
            paddle.rand(shape=[171888, 4], dtype=paddle.float32),
        )
        self.net = LayerCase

    def set_flags(self):
        # NOTE(Aurelius84): cinn_op.pool2d only support pool_type='avg' under adaptive=True
        paddle.set_flags({"FLAGS_deny_cinn_ops": "gather"})


if __name__ == '__main__':
    unittest.main()
