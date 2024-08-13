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
# model: configs^ppyoloe^ppyoloe_crn_l_300e_coco_single_dy2st_train
# api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||method:__sub__||method:__sub__||method:__add__||method:__add__||api:paddle.tensor.manipulation.stack||method:astype||api:paddle.tensor.manipulation.stack||method:astype||method:reshape||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||method:__sub__||method:__sub__||method:__add__||method:__add__||api:paddle.tensor.manipulation.stack||method:astype||api:paddle.tensor.manipulation.stack||method:astype||method:reshape||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||method:__sub__||method:__sub__||method:__add__||method:__add__||api:paddle.tensor.manipulation.stack||method:astype||api:paddle.tensor.manipulation.stack||method:astype||method:reshape||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat
from base import *  # noqa: F403


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
    ):
        var_0 = paddle.tensor.creation.arange(end=11)
        var_1 = var_0 + 0.5
        var_2 = var_1 * 32
        var_3 = paddle.tensor.creation.arange(end=11)
        var_4 = var_3 + 0.5
        var_5 = var_4 * 32
        var_6, var_7 = paddle.tensor.creation.meshgrid(var_5, var_2)
        var_8 = var_7 - 80.0
        var_9 = var_6 - 80.0
        var_10 = var_7 + 80.0
        var_11 = var_6 + 80.0
        var_12 = paddle.tensor.manipulation.stack(
            [var_8, var_9, var_10, var_11], axis=-1
        )
        var_13 = var_12.astype('float32')
        var_14 = paddle.tensor.manipulation.stack([var_7, var_6], axis=-1)
        var_15 = var_14.astype('float32')
        var_16 = var_13.reshape([-1, 4])
        var_17 = var_15.reshape([-1, 2])
        var_18 = paddle.tensor.creation.full([121, 1], 32, dtype='float32')
        var_19 = paddle.tensor.creation.arange(end=22)
        var_20 = var_19 + 0.5
        var_21 = var_20 * 16
        var_22 = paddle.tensor.creation.arange(end=22)
        var_23 = var_22 + 0.5
        var_24 = var_23 * 16
        var_25, var_26 = paddle.tensor.creation.meshgrid(var_24, var_21)
        var_27 = var_26 - 40.0
        var_28 = var_25 - 40.0
        var_29 = var_26 + 40.0
        var_30 = var_25 + 40.0
        var_31 = paddle.tensor.manipulation.stack(
            [var_27, var_28, var_29, var_30], axis=-1
        )
        var_32 = var_31.astype('float32')
        var_33 = paddle.tensor.manipulation.stack([var_26, var_25], axis=-1)
        var_34 = var_33.astype('float32')
        var_35 = var_32.reshape([-1, 4])
        var_36 = var_34.reshape([-1, 2])
        var_37 = paddle.tensor.creation.full([484, 1], 16, dtype='float32')
        var_38 = paddle.tensor.creation.arange(end=44)
        var_39 = var_38 + 0.5
        var_40 = var_39 * 8
        var_41 = paddle.tensor.creation.arange(end=44)
        var_42 = var_41 + 0.5
        var_43 = var_42 * 8
        var_44, var_45 = paddle.tensor.creation.meshgrid(var_43, var_40)
        var_46 = var_45 - 20.0
        var_47 = var_44 - 20.0
        var_48 = var_45 + 20.0
        var_49 = var_44 + 20.0
        var_50 = paddle.tensor.manipulation.stack(
            [var_46, var_47, var_48, var_49], axis=-1
        )
        var_51 = var_50.astype('float32')
        var_52 = paddle.tensor.manipulation.stack([var_45, var_44], axis=-1)
        var_53 = var_52.astype('float32')
        var_54 = var_51.reshape([-1, 4])
        var_55 = var_53.reshape([-1, 2])
        var_56 = paddle.tensor.creation.full([1936, 1], 8, dtype='float32')
        var_57 = paddle.tensor.manipulation.concat([var_16, var_35, var_54])
        var_58 = paddle.tensor.manipulation.concat([var_17, var_36, var_55])
        var_59 = paddle.tensor.manipulation.concat([var_18, var_37, var_56])
        return var_57, var_58, var_16, var_35, var_54, var_59


class TestLayer(TestBase):
    def init(self):
        self.input_specs = []
        self.inputs = ()
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
