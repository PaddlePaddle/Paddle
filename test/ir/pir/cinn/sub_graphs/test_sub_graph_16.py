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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^LeViT^LeViT_128
# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.split||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.concat||api:paddle.tensor.linalg.transpose||method:reshape||api:paddle.tensor.linalg.matmul||method:__mul__||method:__add__||api:paddle.nn.functional.activation.softmax||api:paddle.tensor.linalg.matmul||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[12, 16],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [10, 16, 768], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [16, 16], dtype: paddle.int64, stop_gradient: True)
    ):
        var_2 = paddle.tensor.manipulation.reshape(var_0, [10, 16, 12, 64])
        out = paddle.tensor.manipulation.split(var_2, [16, 16, 32], axis=3)
        var_3 = out[0]
        var_4 = out[1]
        var_5 = out[2]
        var_6 = paddle.tensor.linalg.transpose(var_3, perm=[0, 2, 1, 3])
        var_7 = paddle.tensor.linalg.transpose(var_4, perm=[0, 2, 1, 3])
        var_8 = paddle.tensor.linalg.transpose(var_5, perm=[0, 2, 1, 3])
        var_9 = paddle.tensor.linalg.transpose(var_7, perm=[0, 1, 3, 2])
        var_10 = paddle.tensor.linalg.transpose(
            self.parameter_0,
            (
                1,
                0,
            ),
        )
        concat_list = []
        for i in range(len(var_1)):
            _var = var_1[i]
            var_10 = paddle.tensor.manipulation.gather(var_10, _var)
            concat_list.append(var_10)
        var_43 = paddle.tensor.manipulation.concat(concat_list)
        var_44 = paddle.tensor.linalg.transpose(
            var_43,
            (
                1,
                0,
            ),
        )
        var_45 = var_44.reshape(
            (
                0,
                16,
                16,
            )
        )
        var_46 = paddle.tensor.linalg.matmul(var_6, var_9)
        var_47 = var_46.__mul__(0.25)
        var_48 = var_47.__add__(var_45)
        var_49 = paddle.nn.functional.activation.softmax(var_48)
        var_50 = paddle.tensor.linalg.matmul(var_49, var_8)
        var_51 = paddle.tensor.linalg.transpose(var_50, perm=[0, 2, 1, 3])
        var_52 = paddle.tensor.manipulation.reshape(var_51, [10, 16, 384])
        return var_52


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[10, 16, 768], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[16, 16], dtype=paddle.int64),
        )
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            # NOTE: This Test Can Not Pass with atol 1e-8
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
