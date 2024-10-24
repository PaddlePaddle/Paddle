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
# method:__getitem__||method:__truediv__||method:__getitem__||method:__truediv__||method:__ne__
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [2], dtype: paddle.int64, stop_gradient: True)
    ):
        var_1 = var_0[0]
        var_2 = var_1 / 640
        var_3 = var_0[1]
        var_4 = var_3 / 640
        var_5 = var_4 != 1
        return var_5, var_2, var_4


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1,), dtype=paddle.int64, name=None, stop_gradient=True
            )
        ]
        self.inputs = (
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int64),
        )
        self.net = LayerCase
        self.with_train = True


if __name__ == '__main__':
    unittest.main()
