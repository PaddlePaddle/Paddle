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
# model: configs^mot^jde^jde_darknet53_30e_576x320_single_dy2st_train
# api:paddle.tensor.creation.to_tensor||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__mul__||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__mul__||method:__add__||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__mul__||method:__add__||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__mul__||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__mul__||method:__add__||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__mul__||method:__add__||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__mul__||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__mul__||method:__add__||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__mul__||method:__add__||method:__radd__||method:__add__||method:__add__||method:__radd__||method:__add__||method:__add__||method:__radd__||method:__add__||method:__add__||method:__radd__||method:__add__||method:__add__
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_4,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_5,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_6,  # (shape: [1], dtype: paddle.float32, stop_gradient: False)
        var_7,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_8,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_9 = paddle.tensor.creation.to_tensor(26.0, dtype='float32')
        var_10 = -self.parameter_3
        var_11 = paddle.tensor.ops.exp(var_10)
        var_12 = var_11 * var_0
        var_13 = var_12 + self.parameter_3
        var_14 = var_13 * 0.5
        var_15 = -self.parameter_2
        var_16 = paddle.tensor.ops.exp(var_15)
        var_17 = var_16 * var_3
        var_18 = var_17 + self.parameter_2
        var_19 = var_18 * 0.5
        var_20 = var_14 + var_19
        var_21 = -self.parameter_4
        var_22 = paddle.tensor.ops.exp(var_21)
        var_23 = var_22 * var_6
        var_24 = var_23 + self.parameter_4
        var_25 = var_24 * 0.5
        var_26 = var_20 + var_25
        var_27 = -self.parameter_7
        var_28 = paddle.tensor.ops.exp(var_27)
        var_29 = var_28 * var_1
        var_30 = var_29 + self.parameter_7
        var_31 = var_30 * 0.5
        var_32 = -self.parameter_8
        var_33 = paddle.tensor.ops.exp(var_32)
        var_34 = var_33 * var_4
        var_35 = var_34 + self.parameter_8
        var_36 = var_35 * 0.5
        var_37 = var_31 + var_36
        var_38 = -self.parameter_5
        var_39 = paddle.tensor.ops.exp(var_38)
        var_40 = var_39 * var_7
        var_41 = var_40 + self.parameter_5
        var_42 = var_41 * 0.5
        var_43 = var_37 + var_42
        var_44 = -self.parameter_0
        var_45 = paddle.tensor.ops.exp(var_44)
        var_46 = var_45 * var_2
        var_47 = var_46 + self.parameter_0
        var_48 = var_47 * 0.5
        var_49 = -self.parameter_1
        var_50 = paddle.tensor.ops.exp(var_49)
        var_51 = var_50 * var_5
        var_52 = var_51 + self.parameter_1
        var_53 = var_52 * 0.5
        var_54 = var_48 + var_53
        var_55 = -self.parameter_6
        var_56 = paddle.tensor.ops.exp(var_55)
        var_57 = var_56 * var_8
        var_58 = var_57 + self.parameter_6
        var_59 = var_58 * 0.5
        var_60 = var_54 + var_59
        var_61 = 0 + var_0
        var_62 = var_61 + var_1
        var_63 = var_62 + var_2
        var_64 = 0 + var_3
        var_65 = var_64 + var_4
        var_66 = var_65 + var_5
        var_67 = 0 + var_6
        var_68 = var_67 + var_7
        var_69 = var_68 + var_8
        var_70 = 0 + var_26
        var_71 = var_70 + var_43
        var_72 = var_71 + var_60
        return var_63, var_66, var_69, var_72, var_9


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
