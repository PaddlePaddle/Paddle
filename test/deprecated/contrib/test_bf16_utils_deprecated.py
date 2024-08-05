# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base
from paddle.base import core
from paddle.static import amp

paddle.enable_static()


class AMPTest2(unittest.TestCase):
    def test_find_op_index(self):
        block = base.default_main_program().global_block()
        op_desc = core.OpDesc()
        idx = amp.fp16_utils.find_op_index(block.desc, op_desc)
        assert idx == -1

    def test_is_in_fp32_varnames(self):
        block = base.default_main_program().global_block()

        var1 = block.create_var(name="X", shape=[3], dtype='float32')
        var2 = block.create_var(name="Y", shape=[3], dtype='float32')
        var3 = block.create_var(name="Z", shape=[3], dtype='float32')
        op1 = block.append_op(
            type="abs", inputs={"X": [var1]}, outputs={"Out": [var2]}
        )
        op2 = block.append_op(
            type="abs", inputs={"X": [var2]}, outputs={"Out": [var3]}
        )
        amp_lists_1 = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_varnames={'X'}
        )
        assert amp.bf16.amp_utils._is_in_fp32_varnames(op1, amp_lists_1)
        amp_lists_2 = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_varnames={'Y'}
        )
        assert amp.bf16.amp_utils._is_in_fp32_varnames(op2, amp_lists_2)
        assert amp.bf16.amp_utils._is_in_fp32_varnames(op1, amp_lists_2)

    def test_find_true_post_op(self):
        block = base.default_main_program().global_block()

        var1 = block.create_var(name="X", shape=[3], dtype='float32')
        var2 = block.create_var(name="Y", shape=[3], dtype='float32')
        var3 = block.create_var(name="Z", shape=[3], dtype='float32')
        op1 = block.append_op(
            type="abs", inputs={"X": [var1]}, outputs={"Out": [var2]}
        )
        op2 = block.append_op(
            type="abs", inputs={"X": [var2]}, outputs={"Out": [var3]}
        )
        res = amp.bf16.amp_utils.find_true_post_op(block.ops, op1, "Y")
        assert res == [op2]

    def test_find_true_post_op_with_search_all(self):
        program = base.Program()
        block = program.current_block()
        startup_block = base.default_startup_program().global_block()

        var1 = block.create_var(name="X", shape=[3], dtype='float32')
        var2 = block.create_var(name="Y", shape=[3], dtype='float32')
        initializer_op = startup_block._prepend_op(
            type="fill_constant",
            outputs={"Out": var1},
            attrs={"shape": var1.shape, "dtype": var1.dtype, "value": 1.0},
        )

        op1 = block.append_op(
            type="abs", inputs={"X": [var1]}, outputs={"Out": [var2]}
        )
        result = amp.bf16.amp_utils.find_true_post_op(
            block.ops, initializer_op, "X", search_all=False
        )
        assert len(result) == 0
        result = amp.bf16.amp_utils.find_true_post_op(
            block.ops, initializer_op, "X", search_all=True
        )
        assert result == [op1]


if __name__ == '__main__':
    unittest.main()
