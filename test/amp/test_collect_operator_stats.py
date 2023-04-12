# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class TestOpStatsEager(unittest.TestCase):
    def _check_result(self, dtype):
        # Returned the dict.
        op_list = paddle.fluid.core.get_low_precision_op_list()

        self.assertTrue('elementwise_add' in op_list)
        self.assertTrue('conv2d' in op_list)

        conv2d_called = op_list['conv2d'].split(',')
        add_called = op_list['elementwise_add'].split(',')
        add_num = 0
        conv_num = 0
        for i in range(4):
            add_num += int(add_called[i])
            conv_num += int(add_called[i])

        self.assertTrue(conv_num == 1)
        self.assertTrue(add_num == 1)

        if dtype == "float16":
            self.assertTrue(int(conv2d_called[0]) == 1)
            self.assertTrue(int(add_called[0]) == 1)

    def test_enable_disable(self):
        conv = paddle.nn.Conv2D(3, 2, 3)
        x = paddle.rand([10, 3, 32, 32])

        paddle.amp.debugging.enable_operator_stats_collection()
        # amp list conv2d, elementwise_add, cast (transfer_dtype)
        with paddle.amp.auto_cast(enable=True, level='O2'):
            out = conv(x)
        # Print to the standard output.
        paddle.amp.debugging.disable_operator_stats_collection()

        self._check_result(dtype=out.dtype)

    def test_context(self):
        conv = paddle.nn.Conv2D(3, 2, 3)
        x = paddle.rand([10, 3, 32, 32])

        with paddle.amp.debugging.collect_operator_stats():
            # amp list conv2d, elementwise_add, cast (transfer_dtype)
            with paddle.amp.auto_cast(enable=True, level='O2'):
                out = conv(x)

        self._check_result(dtype=out.dtype)


class WhileNet(nn.Layer):
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


class TestOpStatsStatic(unittest.TestCase):
    def test_while_op(self):
        paddle.enable_static()
        model = WhileNet()
        paddle.static.amp.debugging.collect_operator_stats(
            program=None, print_subblocks=True
        )
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
