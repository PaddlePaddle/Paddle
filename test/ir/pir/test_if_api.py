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
from paddle.base.libpaddle.pir import get_used_external_value

paddle.enable_static()


def true_func():
    a = paddle.full(shape=[1, 2], dtype='float32', fill_value=1)
    b = paddle.full(shape=[2, 3], dtype='int64', fill_value=1)
    return a, b


def false_func():
    a = paddle.full(shape=[1, 2], dtype='float32', fill_value=3)
    b = paddle.full(shape=[2, 3], dtype='int64', fill_value=2)
    return a, b


class TestBuildModuleWithIfOp(unittest.TestCase):
    def test_if_with_single_output(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
            y = paddle.static.data(name="y", shape=[6, 1], dtype="float32")
            out = paddle.static.nn.cond(x < y, lambda: x + y, lambda: x - y)
        if_op = out[0].get_defining_op()
        self.assertEqual(if_op.name(), "pd_op.if")
        self.assertEqual(len(out), 1)
        value_list = get_used_external_value(if_op)
        print(value_list)

    def test_if_with_multiple_output(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
            y = paddle.static.data(name="y", shape=[6, 1], dtype="float32")
            pred = paddle.less_than(x=x, y=y, name=None)
            out = paddle.static.nn.cond(pred, true_func, false_func)
        self.assertEqual(out[0].get_defining_op().name(), "pd_op.if")
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
