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
# api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.manipulation.concat||method:astype||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.split||method:__truediv__||method:__add__||api:paddle.tensor.ops.exp||method:__mul__||method:__sub__||method:__add__||api:paddle.tensor.manipulation.concat||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||api:paddle.tensor.manipulation.stack||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.manipulation.concat||method:astype||api:paddle.tensor.manipulation.concat
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 10000, 80], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 2500, 80], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 625, 80], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [1, 10000, 4], dtype: paddle.float32, stop_gradient: False)
        var_4,  # (shape: [1, 2500, 4], dtype: paddle.float32, stop_gradient: False)
        var_5,  # (shape: [1, 625, 4], dtype: paddle.float32, stop_gradient: False)
        var_6,  # (shape: [1, 10000, 1], dtype: paddle.float32, stop_gradient: False)
        var_7,  # (shape: [1, 2500, 1], dtype: paddle.float32, stop_gradient: False)
        var_8,  # (shape: [1, 625, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_9 = paddle.tensor.manipulation.concat([var_0, var_1, var_2], axis=1)
        var_10 = paddle.tensor.manipulation.concat(
            [var_3, var_4, var_5], axis=1
        )
        var_11 = paddle.tensor.manipulation.concat(
            [var_6, var_7, var_8], axis=1
        )
        var_12 = paddle.tensor.creation.arange(100)
        var_13 = var_12 + 0.0
        var_14 = var_13 * 8
        var_15 = paddle.tensor.creation.arange(100)
        var_16 = var_15 + 0.0
        var_17 = var_16 * 8
        var_18, var_19 = paddle.tensor.creation.meshgrid(var_17, var_14)
        var_20 = paddle.tensor.manipulation.stack([var_19, var_18], axis=-1)
        var_21 = var_20.reshape([-1, 2])
        var_22 = paddle.tensor.creation.full([10000, 1], 8, dtype='float32')
        var_23 = paddle.tensor.creation.arange(50)
        var_24 = var_23 + 0.0
        var_25 = var_24 * 16
        var_26 = paddle.tensor.creation.arange(50)
        var_27 = var_26 + 0.0
        var_28 = var_27 * 16
        var_29, var_30 = paddle.tensor.creation.meshgrid(var_28, var_25)
        var_31 = paddle.tensor.manipulation.stack([var_30, var_29], axis=-1)
        var_32 = var_31.reshape([-1, 2])
        var_33 = paddle.tensor.creation.full([2500, 1], 16, dtype='float32')
        var_34 = paddle.tensor.creation.arange(25)
        var_35 = var_34 + 0.0
        var_36 = var_35 * 32
        var_37 = paddle.tensor.creation.arange(25)
        var_38 = var_37 + 0.0
        var_39 = var_38 * 32
        var_40, var_41 = paddle.tensor.creation.meshgrid(var_39, var_36)
        var_42 = paddle.tensor.manipulation.stack([var_41, var_40], axis=-1)
        var_43 = var_42.reshape([-1, 2])
        var_44 = paddle.tensor.creation.full([625, 1], 32, dtype='float32')
        var_45 = paddle.tensor.manipulation.concat([var_21, var_32, var_43])
        var_46 = var_45.astype('float32')
        var_47 = paddle.tensor.manipulation.concat([var_22, var_33, var_44])
        var_48, var_49 = paddle.tensor.manipulation.split(var_10, 2, axis=-1)
        var_50 = var_46 / var_47
        var_51 = var_48 + var_50
        var_52 = paddle.tensor.ops.exp(var_49)
        var_53 = var_52 * 0.5
        var_54 = var_51 - var_53
        var_55 = var_51 + var_53
        var_56 = paddle.tensor.manipulation.concat([var_54, var_55], axis=-1)
        var_57 = paddle.tensor.creation.arange(100)
        var_58 = var_57 + 0.5
        var_59 = var_58 * 8
        var_60 = paddle.tensor.creation.arange(100)
        var_61 = var_60 + 0.5
        var_62 = var_61 * 8
        var_63, var_64 = paddle.tensor.creation.meshgrid(var_62, var_59)
        var_65 = paddle.tensor.manipulation.stack([var_64, var_63], axis=-1)
        var_66 = var_65.reshape([-1, 2])
        var_67 = paddle.tensor.creation.full([10000, 1], 8, dtype='float32')
        var_68 = paddle.tensor.creation.arange(50)
        var_69 = var_68 + 0.5
        var_70 = var_69 * 16
        var_71 = paddle.tensor.creation.arange(50)
        var_72 = var_71 + 0.5
        var_73 = var_72 * 16
        var_74, var_75 = paddle.tensor.creation.meshgrid(var_73, var_70)
        var_76 = paddle.tensor.manipulation.stack([var_75, var_74], axis=-1)
        var_77 = var_76.reshape([-1, 2])
        var_78 = paddle.tensor.creation.full([2500, 1], 16, dtype='float32')
        var_79 = paddle.tensor.creation.arange(25)
        var_80 = var_79 + 0.5
        var_81 = var_80 * 32
        var_82 = paddle.tensor.creation.arange(25)
        var_83 = var_82 + 0.5
        var_84 = var_83 * 32
        var_85, var_86 = paddle.tensor.creation.meshgrid(var_84, var_81)
        var_87 = paddle.tensor.manipulation.stack([var_86, var_85], axis=-1)
        var_88 = var_87.reshape([-1, 2])
        var_89 = paddle.tensor.creation.full([625, 1], 32, dtype='float32')
        var_90 = paddle.tensor.manipulation.concat([var_66, var_77, var_88])
        var_91 = var_90.astype('float32')
        var_92 = paddle.tensor.manipulation.concat([var_67, var_78, var_89])
        return var_9, var_56, var_11, var_91, var_92, var_66, var_77, var_88


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
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1, 10000, 80], dtype=paddle.float32),
            paddle.rand(shape=[1, 2500, 80], dtype=paddle.float32),
            paddle.rand(shape=[1, 625, 80], dtype=paddle.float32),
            paddle.rand(shape=[1, 10000, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 2500, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 625, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 10000, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 2500, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 625, 1], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.atol = 1e-5


if __name__ == '__main__':
    unittest.main()
