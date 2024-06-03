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
# api:paddle.tensor.manipulation.gather||method:__eq__||api:paddle.tensor.creation.ones_like||method:__mul__||api:paddle.tensor.search.where||method:__eq__||api:paddle.tensor.creation.ones_like||method:__mul__||api:paddle.tensor.search.where
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [2002], dtype: paddle.int64, stop_gradient: True)
        var_1,  # (shape: [2002], dtype: paddle.int32, stop_gradient: True)
        var_2,  # (shape: [2], dtype: paddle.int32, stop_gradient: True)
    ):
        var_3 = paddle.tensor.manipulation.gather(var_2, var_0)
        var_4 = var_1 == 0
        var_5 = paddle.tensor.creation.ones_like(var_3)
        var_6 = var_5 * 80
        var_7 = paddle.tensor.search.where(var_4, var_6, var_3)
        var_8 = var_1 == -1
        var_9 = paddle.tensor.creation.ones_like(var_7)
        var_10 = var_9 * -1
        var_11 = paddle.tensor.search.where(var_8, var_10, var_7)
        return var_11


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1,), dtype=paddle.int64, name=None, stop_gradient=True
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.int32, name=None, stop_gradient=True
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.int32, name=None, stop_gradient=True
            ),
        ]
        self.inputs = (
            paddle.randint(low=0, high=10, shape=[2002], dtype=paddle.int64),
            paddle.randint(low=0, high=10, shape=[2002], dtype=paddle.int32),
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
