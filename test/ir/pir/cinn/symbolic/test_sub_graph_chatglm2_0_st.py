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

# repo: llm_sub_graphs
# model: chatglm2
# method:astype||method:pow||method:mean||method:__add__||api:paddle.tensor.ops.rsqrt||method:__mul__||method:__mul__||method:astype||api:paddle.nn.functional.common.linear||method:split||method:reshape||method:reshape||method:reshape||method:__getitem__||method:__getitem__||method:__getitem__||method:reshape||method:reshape||method:__getitem__||method:__getitem__||method:__mul__||method:__getitem__||method:__getitem__||method:__mul__||method:__sub__||method:__getitem__||method:__getitem__||method:__mul__||method:__getitem__||method:__getitem__||method:__mul__||method:__add__||api:paddle.tensor.manipulation.stack||method:flatten||api:paddle.tensor.manipulation.concat||method:__getitem__||method:__getitem__||method:__getitem__||method:reshape||method:reshape||method:__getitem__||method:__getitem__||method:__mul__||method:__getitem__||method:__getitem__||method:__mul__||method:__sub__||method:__getitem__||method:__getitem__||method:__mul__||method:__getitem__||method:__getitem__||method:__mul__||method:__add__||api:paddle.tensor.manipulation.stack||method:flatten||api:paddle.tensor.manipulation.concat||method:unsqueeze||method:tile||method:reshape||method:unsqueeze||method:tile||method:reshape||method:reshape||method:reshape||method:transpose||method:transpose||api:paddle.tensor.linalg.bmm||method:__mul__||method:reshape||method:astype||method:__mul__||method:__add__||method:astype||api:paddle.nn.functional.activation.softmax||method:astype||api:paddle.nn.functional.common.dropout||method:reshape||method:reshape||method:transpose||api:paddle.tensor.linalg.bmm||method:reshape||method:transpose||method:reshape||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout||method:__add__||method:astype||method:pow||method:mean||method:__add__||api:paddle.tensor.ops.rsqrt||method:__mul__||method:__mul__||method:astype||api:paddle.nn.functional.common.linear||method:__getitem__||method:__getitem__||api:paddle.nn.functional.activation.silu||method:__mul__||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout||method:__add__
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1024, 32],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[32, 32],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[32, 2048],
            dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
            shape=[32, 64],
            dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1024, 4, 32], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [4, 1, 1024, 1024], dtype: paddle.float32, stop_gradient: True)
        var_2,  # (shape: [1024, 1, 2, 2], dtype: paddle.float32, stop_gradient: True)
    ):
        var_3 = var_0.astype('float32')
        var_4 = var_3.pow(2)
        var_5 = var_4.mean(-1, keepdim=True)
        var_6 = var_5 + 1e-05
        var_7 = paddle.tensor.ops.rsqrt(var_6)
        var_8 = var_7 * var_0
        var_9 = var_8 * self.parameter_0
        var_10 = var_9.astype('float32')
        var_11 = paddle.nn.functional.common.linear(
            x=var_10, weight=self.parameter_5, bias=self.parameter_6, name=None
        )
        var_12, var_13, var_14 = var_11.split([32, 16, 16], axis=-1)
        var_15 = var_12.reshape([1024, 4, 4, 8])
        var_16 = var_13.reshape([1024, 4, -1, 8])
        var_17 = var_14.reshape([1024, 4, -1, 8])
        var_18 = var_15[(..., slice(None, 4, None))]
        var_19 = var_15[(..., slice(4, None, None))]
        var_20 = var_2[slice(None, 1024, None)]
        var_21 = var_18.reshape([1024, -1, 4, 2, 2])
        var_22 = var_20.reshape([1024, -1, 1, 2, 2])
        var_23 = var_21[(..., 0)]
        var_24 = var_22[(..., 0)]
        var_25 = var_23 * var_24
        var_26 = var_21[(..., 1)]
        var_27 = var_22[(..., 1)]
        var_28 = var_26 * var_27
        var_29 = var_25 - var_28
        var_30 = var_21[(..., 1)]
        var_31 = var_22[(..., 0)]
        var_32 = var_30 * var_31
        var_33 = var_21[(..., 0)]
        var_34 = var_22[(..., 1)]
        var_35 = var_33 * var_34
        var_36 = var_32 + var_35
        var_37 = paddle.tensor.manipulation.stack([var_29, var_36], -1)
        var_38 = var_37.flatten(3)
        var_39 = paddle.tensor.manipulation.concat(
            (var_38, var_19),
            axis=-1,
        )
        var_40 = var_16[(..., slice(None, 4, None))]
        var_41 = var_16[(..., slice(4, None, None))]
        var_42 = var_2[slice(None, 1024, None)]
        var_43 = var_40.reshape([1024, -1, 2, 2, 2])
        var_44 = var_42.reshape([1024, -1, 1, 2, 2])
        var_45 = var_43[(..., 0)]
        var_46 = var_44[(..., 0)]
        var_47 = var_45 * var_46
        var_48 = var_43[(..., 1)]
        var_49 = var_44[(..., 1)]
        var_50 = var_48 * var_49
        var_51 = var_47 - var_50
        var_52 = var_43[
            (
                ...,
                1,
            )
        ]
        var_53 = var_44[(..., 0)]
        var_54 = var_52 * var_53
        var_55 = var_43[(..., 0)]
        var_56 = var_44[(..., 1)]
        var_57 = var_55 * var_56
        var_58 = var_54 + var_57
        var_59 = paddle.tensor.manipulation.stack([var_51, var_58], -1)
        var_60 = var_59.flatten(3)
        var_61 = paddle.tensor.manipulation.concat(
            (var_60, var_41),
            axis=-1,
        )
        var_62 = var_61.unsqueeze(-2)
        var_63 = var_62.tile([1, 1, 1, 2, 1])
        var_64 = var_63.reshape([1024, 4, 4, 8])
        var_65 = var_17.unsqueeze(-2)
        var_66 = var_65.tile([1, 1, 1, 2, 1])
        var_67 = var_66.reshape([1024, 4, 4, 8])
        var_68 = var_39.reshape([1024, 16, -1])
        var_69 = var_64.reshape([1024, 16, -1])
        var_70 = var_68.transpose([1, 0, 2])
        var_71 = var_69.transpose([1, 2, 0])
        var_72 = paddle.tensor.linalg.bmm(var_70, var_71)
        var_73 = var_72 * 0.01860807318911967
        var_74 = var_73.reshape((4, 4, 1024, 1024))
        var_75 = var_74.astype('float32')
        var_76 = var_75 * 19
        var_77 = var_76 + var_1
        var_78 = var_77.astype('float32')
        var_79 = paddle.nn.functional.activation.softmax(var_78, axis=-1)
        var_80 = var_79.astype('float32')
        var_81 = paddle.nn.functional.common.dropout(
            var_80,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_82 = var_67.reshape([1024, 16, -1])
        var_83 = var_81.reshape([16, 1024, -1])
        var_84 = var_82.transpose([1, 0, 2])
        var_85 = paddle.tensor.linalg.bmm(var_83, var_84)
        var_86 = var_85.reshape((4, 4, 1024, 8))
        var_87 = var_86.transpose([2, 0, 1, 3])
        var_88 = var_87.reshape([1024, 4, 32])
        var_89 = paddle.nn.functional.common.linear(
            x=var_88, weight=self.parameter_2, bias=None, name=None
        )
        var_90 = paddle.nn.functional.common.dropout(
            var_89, p=0.0, training=True
        )
        var_91 = var_0 + var_90
        var_92 = var_91.astype('float32')
        var_93 = var_92.pow(2)
        var_94 = var_93.mean(-1, keepdim=True)
        var_95 = var_94 + 1e-05
        var_96 = paddle.tensor.ops.rsqrt(var_95)
        var_97 = var_96 * var_91
        var_98 = var_97 * self.parameter_4
        var_99 = var_98.astype('float32')
        var_100 = paddle.nn.functional.common.linear(
            x=var_99, weight=self.parameter_3, bias=None, name=None
        )
        var_101 = var_100[(..., slice(None, 1024, None))]
        var_102 = var_100[(..., slice(1024, None, None))]
        var_103 = paddle.nn.functional.activation.silu(var_101)
        var_104 = var_103 * var_102
        var_105 = paddle.nn.functional.common.linear(
            x=var_104, weight=self.parameter_1, bias=None, name=None
        )
        var_106 = paddle.nn.functional.common.dropout(
            var_105, p=0.0, training=True
        )
        var_107 = var_91 + var_106
        return var_107, var_61, var_17


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1024, 4, 32], dtype=paddle.float32),
        paddle.rand(shape=[4, 1, 1024, 1024], dtype=paddle.float32),
        paddle.rand(shape=[1024, 1, 2, 2], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1024, 4, 32]).astype('float32'),
        np.random.random(size=[4, 1, 1024, 1024]).astype('float32'),
        np.random.random(size=[1024, 1, 2, 2]).astype('float32'),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_paddle_inputs()
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

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=False, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
