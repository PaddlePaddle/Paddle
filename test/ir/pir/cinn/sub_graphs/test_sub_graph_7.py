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
# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.split||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||api:paddle.tensor.linalg.transpose||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||method:__getitem__||api:paddle.tensor.manipulation.gather||api:paddle.tensor.manipulation.concat||api:paddle.tensor.linalg.transpose||method:reshape||api:paddle.tensor.linalg.matmul||method:__mul__||method:__add__||api:paddle.nn.functional.activation.softmax||api:paddle.tensor.linalg.matmul||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[8, 49],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [22, 49, 512], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [49, 49], dtype: paddle.int64, stop_gradient: True)
    ):
        var_2 = paddle.tensor.manipulation.reshape(var_0, [22, 49, 8, 64])
        var_3, var_4, var_5 = paddle.tensor.manipulation.split(
            var_2, [16, 16, 32], axis=3
        )
        var_6 = paddle.tensor.linalg.transpose(var_3, perm=[0, 2, 1, 3])
        var_7 = paddle.tensor.linalg.transpose(var_4, perm=[0, 2, 1, 3])
        var_8 = paddle.tensor.linalg.transpose(var_5, perm=[0, 2, 1, 3])
        var_9 = paddle.tensor.linalg.transpose(var_7, perm=[0, 1, 3, 2])
        var_10 = paddle.tensor.linalg.transpose(self.parameter_0, (1, 0))
        concat_list = []
        for var in var_1:
            concat_list.append(paddle.tensor.manipulation.gather(var_10, var))
        var_11 = paddle.tensor.manipulation.concat(concat_list)
        var_12 = paddle.tensor.linalg.transpose(var_11, (1, 0))
        var_13 = var_12.reshape((0, 49, 49))
        var_14 = paddle.tensor.linalg.matmul(var_6, var_9)
        var_15 = var_14 * 0.25
        var_16 = var_15 + var_13
        var_17 = paddle.nn.functional.activation.softmax(var_16)
        var_18 = paddle.tensor.linalg.matmul(var_17, var_8)
        var_19 = paddle.tensor.linalg.transpose(var_18, perm=[0, 2, 1, 3])
        var_20 = paddle.tensor.manipulation.reshape(var_19, [22, 49, 256])
        return var_20


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
                # TODO(xiaoyao0115): -1 shape in first dim will bring while_op, skip it for now
                shape=(49, -1),
                dtype=paddle.int64,
                name=None,
                stop_gradient=True,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[22, 49, 512], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[49, 49], dtype=paddle.int64),
        )
        self.net = LayerCase

    # NOTE prim + cinn lead to error
    # NOTE can not pass when atol=1e-8 with prim


if __name__ == '__main__':
    unittest.main()
