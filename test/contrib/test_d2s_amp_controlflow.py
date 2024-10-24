# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
from paddle import nn


class Net_Cond(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self):
        cond_input_x = paddle.ones(shape=[32, 32], dtype="float32")
        cond_input_y = paddle.zeros(shape=[32, 32], dtype="float32")
        if paddle.shape(cond_input_x)[0] <= paddle.shape(cond_input_y)[0]:
            cond_input_y = paddle.matmul(
                cond_input_x,
                cond_input_x.T,
            )
        return cond_input_y.mean()


class Net_While(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self):
        while_input_x = paddle.ones(shape=[64, 32], dtype="float32")
        while_input_y = paddle.zeros(shape=[32, 32], dtype="float32")
        while paddle.shape(while_input_x)[1] >= paddle.shape(while_input_y)[1]:
            while_input_y = paddle.matmul(
                while_input_x,
                while_input_x.T,
            )
        return while_input_y.mean()


class Net_Sub_Block_FP32(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self):
        cond_input_x = paddle.ones(shape=[32, 32], dtype="float32")
        cond_input_y = paddle.zeros(shape=[32, 32], dtype="float32")
        if paddle.shape(cond_input_x)[0] <= paddle.shape(cond_input_y)[0]:
            cond_input_y = paddle.log(cond_input_x)
        return cond_input_y.mean()


class TestD2SAmpWithControlFlowOp(unittest.TestCase):
    def test_cond_op(self):
        model = Net_Cond()
        model = paddle.jit.to_static(model, full_graph=True)
        model = paddle.amp.decorate(
            models=model, level='O2', save_dtype="float32"
        )
        with paddle.amp.auto_cast(level='O2'):
            model()

    def test_while_op(self):
        model = Net_While()
        model = paddle.jit.to_static(model, full_graph=True)
        model = paddle.amp.decorate(
            models=model, level='O2', save_dtype="float32"
        )
        with paddle.amp.auto_cast(level='O2'):
            model()

    def test_sub_block_fp32_op(self):
        model = Net_Sub_Block_FP32()
        model = paddle.jit.to_static(model, full_graph=True)
        model = paddle.amp.decorate(
            models=model, level='O2', save_dtype="float32"
        )
        with paddle.amp.auto_cast(level='O2'):
            model()


if __name__ == '__main__':
    unittest.main()
