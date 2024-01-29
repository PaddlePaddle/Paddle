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
import unittest

import numpy as np

import paddle


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
        var_10 = self.parameter_3.__neg__()
        var_11 = paddle.tensor.ops.exp(var_10)
        var_12 = var_11.__mul__(var_0)
        var_13 = var_12.__add__(self.parameter_3)
        var_14 = var_13.__mul__(0.5)
        var_15 = self.parameter_2.__neg__()
        var_16 = paddle.tensor.ops.exp(var_15)
        var_17 = var_16.__mul__(var_3)
        var_18 = var_17.__add__(self.parameter_2)
        var_19 = var_18.__mul__(0.5)
        var_20 = var_14.__add__(var_19)
        var_21 = self.parameter_4.__neg__()
        var_22 = paddle.tensor.ops.exp(var_21)
        var_23 = var_22.__mul__(var_6)
        var_24 = var_23.__add__(self.parameter_4)
        var_25 = var_24.__mul__(0.5)
        var_26 = var_20.__add__(var_25)
        var_27 = self.parameter_7.__neg__()
        var_28 = paddle.tensor.ops.exp(var_27)
        var_29 = var_28.__mul__(var_1)
        var_30 = var_29.__add__(self.parameter_7)
        var_31 = var_30.__mul__(0.5)
        var_32 = self.parameter_8.__neg__()
        var_33 = paddle.tensor.ops.exp(var_32)
        var_34 = var_33.__mul__(var_4)
        var_35 = var_34.__add__(self.parameter_8)
        var_36 = var_35.__mul__(0.5)
        var_37 = var_31.__add__(var_36)
        var_38 = self.parameter_5.__neg__()
        var_39 = paddle.tensor.ops.exp(var_38)
        var_40 = var_39.__mul__(var_7)
        var_41 = var_40.__add__(self.parameter_5)
        var_42 = var_41.__mul__(0.5)
        var_43 = var_37.__add__(var_42)
        var_44 = self.parameter_0.__neg__()
        var_45 = paddle.tensor.ops.exp(var_44)
        var_46 = var_45.__mul__(var_2)
        var_47 = var_46.__add__(self.parameter_0)
        var_48 = var_47.__mul__(0.5)
        var_49 = self.parameter_1.__neg__()
        var_50 = paddle.tensor.ops.exp(var_49)
        var_51 = var_50.__mul__(var_5)
        var_52 = var_51.__add__(self.parameter_1)
        var_53 = var_52.__mul__(0.5)
        var_54 = var_48.__add__(var_53)
        var_55 = self.parameter_6.__neg__()
        var_56 = paddle.tensor.ops.exp(var_55)
        var_57 = var_56.__mul__(var_8)
        var_58 = var_57.__add__(self.parameter_6)
        var_59 = var_58.__mul__(0.5)
        var_60 = var_54.__add__(var_59)
        var_61 = var_0.__radd__(0)
        var_62 = var_61.__add__(var_1)
        var_63 = var_62.__add__(var_2)
        var_64 = var_3.__radd__(0)
        var_65 = var_64.__add__(var_4)
        var_66 = var_65.__add__(var_5)
        var_67 = var_6.__radd__(0)
        var_68 = var_67.__add__(var_7)
        var_69 = var_68.__add__(var_8)
        var_70 = var_26.__radd__(0)
        var_71 = var_70.__add__(var_43)
        var_72 = var_71.__add__(var_60)
        return var_63, var_66, var_69, var_72, var_9


class TestLayer(unittest.TestCase):
    def setUp(self):
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
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    # NOTE prim + cinn lead to error
    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
