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
# model: configs^picodet^legacy_model^picodet_s_320_coco_single_dy2st_train
# api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||method:flatten||method:flatten
from base import *  # noqa: F403


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
    ):
        var_0 = paddle.tensor.creation.arange(16, dtype='float32')
        var_1 = var_0 + 0.5
        var_2 = var_1 * 16
        var_3 = paddle.tensor.creation.arange(16, dtype='float32')
        var_4 = var_3 + 0.5
        var_5 = var_4 * 16
        var_6, var_7 = paddle.tensor.creation.meshgrid(var_5, var_2)
        var_8 = var_6.flatten()
        var_9 = var_7.flatten()
        return var_8, var_9


class TestLayer(TestBase):
    def init(self):
        self.input_specs = []
        self.inputs = ()
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
