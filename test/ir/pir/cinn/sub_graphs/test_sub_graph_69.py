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
# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.concat||method:__eq__||api:paddle.tensor.search.nonzero||method:__ge__||api:paddle.tensor.search.nonzero
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [171888], dtype: paddle.int32, stop_gradient: True)
        var_1,  # (shape: [1, 171888, 1], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 171888, 4], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = paddle.tensor.manipulation.reshape(x=var_1, shape=(-1,))
        var_4 = paddle.tensor.manipulation.reshape(x=var_2, shape=(-1, 4))
        var_5 = paddle.tensor.manipulation.concat([var_0])
        var_6 = var_5 == 1
        var_7 = paddle.tensor.search.nonzero(var_6)
        var_8 = var_5 >= 0
        var_9 = paddle.tensor.search.nonzero(var_8)
        return var_9, var_3, var_5, var_7, var_4


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1,), dtype=paddle.int32, name=None, stop_gradient=True
            ),
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
        ]
        self.inputs = (
            paddle.randint(low=0, high=10, shape=[171888], dtype=paddle.int32),
            paddle.rand(shape=[1, 171888, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 171888, 4], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.with_precision_compare = False


if __name__ == '__main__':
    unittest.main()
