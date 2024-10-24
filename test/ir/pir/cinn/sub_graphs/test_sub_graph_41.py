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
# method:flatten||api:paddle.tensor.ops.sigmoid||method:flatten||api:paddle.tensor.manipulation.concat||method:__gt__||method:all
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 100, 80], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 100, 4], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [2], dtype: paddle.int32, stop_gradient: True)
    ):
        var_3 = var_0.flatten(start_axis=0, stop_axis=1)
        var_4 = paddle.tensor.ops.sigmoid(var_3)
        var_5 = var_1.flatten(start_axis=0, stop_axis=1)
        # TODO(Aurelius84): concat compute logic does not support single element.
        # var_6 = paddle.tensor.manipulation.concat([var_2])
        var_6 = var_2
        var_7 = var_6 > -1
        var_8 = var_7.all()
        return var_8, var_4, var_6, var_5


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.int32, name=None, stop_gradient=True
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1, 100, 80], dtype=paddle.float32),
            paddle.rand(shape=[1, 100, 4], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
