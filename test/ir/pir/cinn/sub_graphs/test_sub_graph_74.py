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
# model: configs^mask_rcnn^mask_rcnn_r101_vd_fpn_1x_coco_single_dy2st_train
# api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [2, 1], dtype: paddle.int64, stop_gradient: True)
        var_1,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
        var_2,  # (shape: [2], dtype: paddle.int32, stop_gradient: True)
        var_3,  # (shape: [2, 28, 28], dtype: paddle.int32, stop_gradient: True)
        var_4,  # (shape: [2], dtype: paddle.float32, stop_gradient: True)
    ):
        var_5 = paddle.tensor.manipulation.concat([var_0])
        var_6 = paddle.tensor.manipulation.concat([var_1])
        var_7 = paddle.tensor.manipulation.concat([var_2], axis=0)
        var_8 = paddle.tensor.manipulation.concat([var_3], axis=0)
        var_9 = paddle.tensor.manipulation.concat([var_4], axis=0)
        return var_6, var_7, var_8, var_5, var_9


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1),
                dtype=paddle.int64,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.int32, name=None, stop_gradient=True
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.int32, name=None, stop_gradient=True
            ),
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.int32,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.float32, name=None, stop_gradient=True
            ),
        ]
        self.inputs = (
            paddle.randint(low=0, high=10, shape=[2, 1], dtype=paddle.int64),
            paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
            paddle.randint(
                low=0, high=10, shape=[2, 28, 28], dtype=paddle.int32
            ),
            paddle.rand(shape=[2], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.with_precision_compare = False


if __name__ == '__main__':
    unittest.main()
