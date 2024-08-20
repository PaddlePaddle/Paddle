# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core
from paddle.static.amp import fp16_utils

paddle.enable_static()


class AMPTest(unittest.TestCase):
    def test_find_op_index(self):
        with paddle.pir_utils.OldIrGuard():
            block = paddle.base.default_main_program().global_block()
            op_desc = core.OpDesc()
            idx = fp16_utils.find_op_index(block.desc, op_desc)
            assert idx == -1

    def test_find_true_post_op(self):
        with paddle.pir_utils.OldIrGuard():
            block = paddle.base.default_main_program().global_block()

            var1 = block.create_var(name="X", shape=[3], dtype='float32')
            var2 = block.create_var(name="Y", shape=[3], dtype='float32')
            var3 = block.create_var(name="Z", shape=[3], dtype='float32')
            op1 = block.append_op(
                type="abs", inputs={"X": [var1]}, outputs={"Out": [var2]}
            )
            op2 = block.append_op(
                type="abs", inputs={"X": [var2]}, outputs={"Out": [var3]}
            )
            res = fp16_utils.find_true_post_op(block.ops, op1, "Y")
            assert res == [op2]


if __name__ == '__main__':
    unittest.main()
