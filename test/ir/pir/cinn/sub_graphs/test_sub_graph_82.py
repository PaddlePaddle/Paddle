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
# method:astype||method:unsqueeze||method:tile||method:astype||api:paddle.tensor.search.masked_select||method:reshape||api:paddle.tensor.search.masked_select||method:reshape||method:sum||api:paddle.tensor.search.masked_select||method:unsqueeze||api:paddle.nn.functional.loss.l1_loss||api:paddle.tensor.manipulation.split||api:paddle.tensor.manipulation.split||api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.minimum||method:__sub__||method:clip||method:__sub__||method:clip||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:__add__||method:__sub__||method:__add__||method:__truediv__||api:paddle.tensor.math.minimum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||method:__sub__||method:__sub__||method:__mul__||method:__add__||method:__sub__||method:__truediv__||method:__sub__||method:__rsub__||method:__mul__||method:__mul__||method:sum||method:__truediv__||method:unsqueeze||method:astype||method:tile||method:astype||api:paddle.tensor.search.masked_select||method:reshape||api:paddle.tensor.manipulation.split||method:__sub__||method:__sub__||api:paddle.tensor.manipulation.concat||method:clip||api:paddle.tensor.search.masked_select||method:reshape||method:floor||api:paddle.tensor.manipulation.cast||method:__add__||method:astype||method:__sub__||method:__rsub__||method:__sub__||api:paddle.nn.functional.loss.cross_entropy||method:__mul__||method:__sub__||api:paddle.nn.functional.loss.cross_entropy||method:__mul__||method:__add__||method:mean||method:__mul__||method:sum||method:__truediv__
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 2541], dtype: paddle.bool, stop_gradient: True)
        var_1,  # (shape: [1, 2541, 4], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 2541, 4], dtype: paddle.float32, stop_gradient: True)
        var_3,  # (shape: [1, 2541, 80], dtype: paddle.float32, stop_gradient: True)
        var_4,  # (shape: [], dtype: paddle.float32, stop_gradient: True)
        var_5,  # (shape: [1, 2541, 68], dtype: paddle.float32, stop_gradient: False)
        var_6,  # (shape: [2541, 2], dtype: paddle.float32, stop_gradient: True)
    ):
        var_7 = var_0.astype('int32')
        var_8 = var_7.unsqueeze(-1)
        var_9 = var_8.tile([1, 1, 4])
        var_10 = var_9.astype('bool')
        var_11 = paddle.tensor.search.masked_select(var_1, var_10)
        var_12 = var_11.reshape([-1, 4])
        var_13 = paddle.tensor.search.masked_select(var_2, var_10)
        var_14 = var_13.reshape([-1, 4])
        var_15 = var_3.sum(-1)
        var_16 = paddle.tensor.search.masked_select(var_15, var_0)
        var_17 = var_16.unsqueeze(-1)
        var_18 = paddle.nn.functional.loss.l1_loss(var_12, var_14)
        var_19, var_20, var_21, var_22 = paddle.tensor.manipulation.split(
            var_12, num_or_sections=4, axis=-1
        )
        var_23, var_24, var_25, var_26 = paddle.tensor.manipulation.split(
            var_14, num_or_sections=4, axis=-1
        )
        var_27 = paddle.tensor.math.maximum(var_19, var_23)
        var_28 = paddle.tensor.math.maximum(var_20, var_24)
        var_29 = paddle.tensor.math.minimum(var_21, var_25)
        var_30 = paddle.tensor.math.minimum(var_22, var_26)
        var_31 = var_29.__sub__(var_27)
        var_32 = var_31.clip(0)
        var_33 = var_30.__sub__(var_28)
        var_34 = var_33.clip(0)
        var_35 = var_32.__mul__(var_34)
        var_36 = var_21.__sub__(var_19)
        var_37 = var_22.__sub__(var_20)
        var_38 = var_36.__mul__(var_37)
        var_39 = var_25.__sub__(var_23)
        var_40 = var_26.__sub__(var_24)
        var_41 = var_39.__mul__(var_40)
        var_42 = var_38.__add__(var_41)
        var_43 = var_42.__sub__(var_35)
        var_44 = var_43.__add__(1e-10)
        var_45 = var_35.__truediv__(var_44)
        var_46 = paddle.tensor.math.minimum(var_19, var_23)
        var_47 = paddle.tensor.math.minimum(var_20, var_24)
        var_48 = paddle.tensor.math.maximum(var_21, var_25)
        var_49 = paddle.tensor.math.maximum(var_22, var_26)
        var_50 = var_48.__sub__(var_46)
        var_51 = var_49.__sub__(var_47)
        var_52 = var_50.__mul__(var_51)
        var_53 = var_52.__add__(1e-10)
        var_54 = var_53.__sub__(var_44)
        var_55 = var_54.__truediv__(var_53)
        var_56 = var_45.__sub__(var_55)
        var_57 = var_56.__rsub__(1)
        var_58 = var_57.__mul__(1.0)
        var_59 = var_58.__mul__(var_17)
        var_60 = var_59.sum()
        var_61 = var_60.__truediv__(var_4)
        var_62 = var_0.unsqueeze(-1)
        var_63 = var_62.astype('int32')
        var_64 = var_63.tile([1, 1, 68])
        var_65 = var_64.astype('bool')
        var_66 = paddle.tensor.search.masked_select(var_5, var_65)
        var_67 = var_66.reshape([-1, 4, 17])
        var_68, var_69 = paddle.tensor.manipulation.split(var_2, 2, -1)
        var_70 = var_6.__sub__(var_68)
        var_71 = var_69.__sub__(var_6)
        var_72 = paddle.tensor.manipulation.concat([var_70, var_71], -1)
        var_73 = var_72.clip(0, 15.99)
        var_74 = paddle.tensor.search.masked_select(var_73, var_10)
        var_75 = var_74.reshape([-1, 4])
        var_76 = var_75.floor()
        var_77 = paddle.tensor.manipulation.cast(var_76, 'int64')
        var_78 = var_77.__add__(1)
        var_79 = var_78.astype('float32')
        var_80 = var_79.__sub__(var_75)
        var_81 = var_80.__rsub__(1)
        var_82 = var_77.__sub__(0)
        var_83 = paddle.nn.functional.loss.cross_entropy(
            var_67, var_82, reduction='none'
        )
        var_84 = var_83.__mul__(var_80)
        var_85 = var_78.__sub__(0)
        var_86 = paddle.nn.functional.loss.cross_entropy(
            var_67, var_85, reduction='none'
        )
        var_87 = var_86.__mul__(var_81)
        var_88 = var_84.__add__(var_87)
        var_89 = var_88.mean(-1, keepdim=True)
        var_90 = var_89.__mul__(var_17)
        var_91 = var_90.sum()
        var_92 = var_91.__truediv__(var_4)
        return var_18, var_61, var_92


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.randint(
                low=0, high=2, shape=[1, 2541], dtype=paddle.int32
            ).cast(paddle.bool),
            paddle.rand(shape=[1, 2541, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 2541, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 2541, 80], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1, 2541, 68], dtype=paddle.float32),
            paddle.rand(shape=[2541, 2], dtype=paddle.float32),
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
            self.net, to_static=True, with_prim=False, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
