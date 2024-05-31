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
# api:paddle.tensor.manipulation.concat||method:__pow__||method:__rmul__||method:__rsub__||method:__add__||method:log||method:__neg__||method:__mul__||method:__rsub__||method:__pow__||method:__rmul__||method:__add__||method:log||method:__neg__||method:__mul__||api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.gather||method:__sub__||method:unsqueeze||api:paddle.tensor.manipulation.concat||method:unsqueeze||method:tile||method:flatten||api:paddle.tensor.manipulation.concat||method:__truediv__||method:__truediv__||method:unsqueeze||api:paddle.nn.functional.loss.l1_loss||method:sum
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [100, 80], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [2], dtype: paddle.int32, stop_gradient: True)
        var_2,  # (shape: [100, 4], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [2, 4], dtype: paddle.float32, stop_gradient: True)
        var_4,  # (shape: [4], dtype: paddle.float32, stop_gradient: True)
        var_5,  # (shape: [2, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        # TODO(Aurelius84): concat compute logic does not support single element.
        var_6 = var_3
        # var_6 = paddle.tensor.manipulation.concat([var_3])
        var_7 = var_0**2.0
        var_8 = 0.75 * var_7
        var_9 = 1 - var_0
        var_10 = var_9 + 1e-08
        var_11 = var_10.log()
        var_12 = -var_11
        var_13 = var_8 * var_12
        var_14 = 1 - var_0
        var_15 = var_14**2.0
        var_16 = 0.25 * var_15
        var_17 = var_0 + 1e-08
        var_18 = var_17.log()
        var_19 = -var_18
        var_20 = var_16 * var_19
        var_21 = paddle.tensor.manipulation.gather(var_20, var_1, axis=1)
        var_22 = paddle.tensor.manipulation.gather(var_13, var_1, axis=1)
        var_23 = var_21 - var_22
        var_24 = var_4.unsqueeze(0)
        var_25 = var_24
        # var_25 = paddle.tensor.manipulation.concat([var_24])
        var_26 = var_25.unsqueeze(1)
        var_27 = var_26.tile([1, 100, 1])
        var_28 = var_27.flatten(start_axis=0, stop_axis=1)
        var_29 = var_5
        # var_29 = paddle.tensor.manipulation.concat([var_5])
        var_30 = var_2 / var_28
        var_31 = var_6 / var_29
        var_32 = var_30.unsqueeze(-2)
        var_33 = paddle.nn.functional.loss.l1_loss(
            var_32, var_31, reduction='none'
        )
        var_34 = var_33.sum(-1)
        return var_6, var_34, var_23


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.int32, name=None, stop_gradient=True
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
            InputSpec(
                shape=(-1,), dtype=paddle.float32, name=None, stop_gradient=True
            ),
            InputSpec(
                shape=(-1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[100, 80], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
            paddle.rand(shape=[100, 4], dtype=paddle.float32),
            paddle.rand(shape=[2, 4], dtype=paddle.float32),
            paddle.rand(shape=[4], dtype=paddle.float32),
            paddle.rand(shape=[2, 4], dtype=paddle.float32),
        )
        self.net = LayerCase

    def set_flags(self):
        # NOTE(Aurelius84): cinn_op.pool2d only support pool_type='avg' under adaptive=True
        paddle.set_flags({"FLAGS_deny_cinn_ops": "gather"})


if __name__ == '__main__':
    unittest.main()
