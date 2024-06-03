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
# model: configs^rotate^ppyoloe_r^ppyoloe_r_crn_s_3x_dota_single_dy2st_train
# api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||api:paddle.tensor.manipulation.cast||method:reshape||method:__mul__||api:paddle.tensor.creation.full||method:__mul__||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||api:paddle.tensor.manipulation.cast||method:reshape||method:__mul__||api:paddle.tensor.creation.full||method:__mul__||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||method:__getitem__||method:__getitem__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||api:paddle.tensor.manipulation.cast||method:reshape||method:__mul__||api:paddle.tensor.creation.full||method:__mul__||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 384, 32, 32], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 192, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 96, 128, 128], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = var_0.shape
        var_6 = var_3[2]
        var_7 = var_3[3]
        var_8 = paddle.tensor.creation.arange(end=var_7)
        var_9 = var_8 + 0.5
        var_10 = var_9 * 32
        var_11 = paddle.tensor.creation.arange(end=var_6)
        var_12 = var_11 + 0.5
        var_13 = var_12 * 32
        var_14, var_15 = paddle.tensor.creation.meshgrid(var_13, var_10)
        var_16 = paddle.tensor.manipulation.stack([var_15, var_14], axis=-1)
        var_17 = paddle.tensor.manipulation.cast(var_16, dtype='float32')
        var_18 = var_17.reshape([1, -1, 2])
        var_19 = var_6 * var_7
        var_20 = paddle.tensor.creation.full(
            [1, var_19, 1], 32, dtype='float32'
        )
        var_21 = var_6 * var_7
        var_22 = var_1.shape
        var_25 = var_22[2]
        var_26 = var_22[3]
        var_27 = paddle.tensor.creation.arange(end=var_26)
        var_28 = var_27 + 0.5
        var_29 = var_28 * 16
        var_30 = paddle.tensor.creation.arange(end=var_25)
        var_31 = var_30 + 0.5
        var_32 = var_31 * 16
        var_33, var_34 = paddle.tensor.creation.meshgrid(var_32, var_29)
        var_35 = paddle.tensor.manipulation.stack([var_34, var_33], axis=-1)
        var_36 = paddle.tensor.manipulation.cast(var_35, dtype='float32')
        var_37 = var_36.reshape([1, -1, 2])
        var_38 = var_25 * var_26
        var_39 = paddle.tensor.creation.full(
            [1, var_38, 1], 16, dtype='float32'
        )
        var_41 = var_2.shape
        var_44 = var_41[2]
        var_45 = var_41[3]
        var_46 = paddle.tensor.creation.arange(end=var_45)
        var_47 = var_46 + 0.5
        var_48 = var_47 * 8
        var_49 = paddle.tensor.creation.arange(end=var_44)
        var_50 = var_49 + 0.5
        var_51 = var_50 * 8
        var_52, var_53 = paddle.tensor.creation.meshgrid(var_51, var_48)
        var_54 = paddle.tensor.manipulation.stack([var_53, var_52], axis=-1)
        var_55 = paddle.tensor.manipulation.cast(var_54, dtype='float32')
        var_56 = var_55.reshape([1, -1, 2])
        var_57 = var_44 * var_45
        var_58 = paddle.tensor.creation.full([1, var_57, 1], 8, dtype='float32')
        var_60 = paddle.tensor.manipulation.concat(
            [var_18, var_37, var_56], axis=1
        )
        var_61 = paddle.tensor.manipulation.concat(
            [var_20, var_39, var_58], axis=1
        )
        return var_60, var_61


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1, 384, 32, 32], dtype=paddle.float32),
            paddle.rand(shape=[1, 192, 64, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 96, 128, 128], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
