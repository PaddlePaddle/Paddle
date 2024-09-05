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
# model: configs^mot^mcfairmot^mcfairmot_dla34_30e_1088x608_visdrone_single_dy2st_train
# api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.unsqueeze||api:paddle.tensor.creation.full||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.gather_nd||api:paddle.tensor.manipulation.unsqueeze||api:paddle.tensor.manipulation.expand_as||method:__gt__||api:paddle.tensor.search.masked_select||api:paddle.tensor.manipulation.reshape
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 128, 152, 272], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 500], dtype: paddle.int64, stop_gradient: True)
        var_2,  # (shape: [1, 500], dtype: paddle.int32, stop_gradient: True)
    ):
        var_3 = paddle.tensor.linalg.transpose(var_0, perm=[0, 2, 3, 1])
        var_4 = paddle.tensor.manipulation.reshape(var_3, shape=[1, -1, 128])
        var_5 = paddle.tensor.manipulation.unsqueeze(var_1, 2)
        var_6 = paddle.tensor.creation.full(
            shape=[1, 500, 1], fill_value=0, dtype='int64'
        )
        # TODO(Aurelius84): CINN doesn't support concat single element.
        # var_7 = paddle.tensor.manipulation.concat([var_6], axis=0)
        var_7 = var_6
        var_8 = paddle.tensor.manipulation.concat(x=[var_7, var_5], axis=2)
        var_9 = paddle.tensor.manipulation.gather_nd(var_4, index=var_8)
        var_10 = paddle.tensor.manipulation.unsqueeze(var_2, axis=2)
        var_11 = paddle.tensor.manipulation.expand_as(var_10, var_9)
        var_12 = var_11 > 0
        # TODO(Aurelius84): masked_select will introduce dynamtic shape, skip it for now.
        # var_13 = paddle.tensor.search.masked_select(var_9, var_12)
        # var_14 = paddle.tensor.manipulation.reshape(var_13, shape=[-1, 128])
        # return var_8, var_14
        var_12 = paddle.tensor.manipulation.cast(var_12, dtype=var_9.dtype)
        return var_9 + var_12


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
                shape=(-1, -1),
                dtype=paddle.int64,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1, -1),
                dtype=paddle.int32,
                name=None,
                stop_gradient=True,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1, 128, 152, 272], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[1, 500], dtype=paddle.int64),
            paddle.randint(low=0, high=10, shape=[1, 500], dtype=paddle.int32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
