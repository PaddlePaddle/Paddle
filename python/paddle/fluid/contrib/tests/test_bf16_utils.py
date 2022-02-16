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
import copy
import unittest
import paddle.fluid as fluid
import paddle.static.amp as amp
from paddle.fluid import core
import paddle

paddle.enable_static()


class AMPTest(unittest.TestCase):
    def setUp(self):
        self.bf16_list = copy.copy(amp.bf16.amp_lists.bf16_list)
        self.fp32_list = copy.copy(amp.bf16.amp_lists.fp32_list)
        self.gray_list = copy.copy(amp.bf16.amp_lists.gray_list)
        self.amp_lists_ = None

    def tearDown(self):
        self.assertEqual(self.amp_lists_.bf16_list, self.bf16_list)
        self.assertEqual(self.amp_lists_.fp32_list, self.fp32_list)
        self.assertEqual(self.amp_lists_.gray_list, self.gray_list)

    def test_amp_lists(self):
        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16()

    def test_amp_lists_1(self):
        # 1. w={'exp}, b=None
        self.bf16_list.add('exp')
        self.fp32_list.remove('exp')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16({'exp'})

    def test_amp_lists_2(self):
        # 2. w={'tanh'}, b=None
        self.fp32_list.remove('tanh')
        self.bf16_list.add('tanh')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16({'tanh'})

    def test_amp_lists_3(self):
        # 3. w={'lstm'}, b=None
        self.bf16_list.add('lstm')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16({'lstm'})

    def test_amp_lists_4(self):
        # 4. w=None, b={'matmul_v2'}
        self.bf16_list.remove('matmul_v2')
        self.fp32_list.add('matmul_v2')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_list={'matmul_v2'})

    def test_amp_lists_5(self):
        # 5. w=None, b={'matmul_v2'}
        self.fp32_list.add('matmul_v2')
        self.bf16_list.remove('matmul_v2')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_list={'matmul_v2'})

    def test_amp_lists_6(self):
        # 6. w=None, b={'lstm'}
        self.fp32_list.add('lstm')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_list={'lstm'})

    def test_amp_lists_7(self):
        self.fp32_list.add('reshape2')
        self.gray_list.remove('reshape2')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_list={'reshape2'})

    def test_amp_list_8(self):
        self.bf16_list.add('reshape2')
        self.gray_list.remove('reshape2')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_bf16_list={'reshape2'})


class AMPTest2(unittest.TestCase):
    def test_amp_lists_(self):
        # 7. w={'lstm'} b={'lstm'}
        # raise ValueError
        self.assertRaises(ValueError, amp.bf16.AutoMixedPrecisionListsBF16,
                          {'lstm'}, {'lstm'})

    def test_find_op_index(self):
        block = fluid.default_main_program().global_block()
        op_desc = core.OpDesc()
        idx = amp.bf16.amp_utils.find_op_index(block.desc, op_desc)
        assert (idx == -1)

    def test_is_in_fp32_varnames(self):
        block = fluid.default_main_program().global_block()

        var1 = block.create_var(name="X", shape=[3], dtype='float32')
        var2 = block.create_var(name="Y", shape=[3], dtype='float32')
        var3 = block.create_var(name="Z", shape=[3], dtype='float32')
        op1 = block.append_op(
            type="abs", inputs={"X": [var1]}, outputs={"Out": [var2]})
        op2 = block.append_op(
            type="abs", inputs={"X": [var2]}, outputs={"Out": [var3]})
        amp_lists_1 = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_varnames={'X'})
        assert amp.bf16.amp_utils._is_in_fp32_varnames(op1, amp_lists_1)
        amp_lists_2 = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_varnames={'Y'})
        assert amp.bf16.amp_utils._is_in_fp32_varnames(op2, amp_lists_2)
        assert amp.bf16.amp_utils._is_in_fp32_varnames(op1, amp_lists_2)

    def test_find_true_post_op(self):

        block = fluid.default_main_program().global_block()

        var1 = block.create_var(name="X", shape=[3], dtype='float32')
        var2 = block.create_var(name="Y", shape=[3], dtype='float32')
        var3 = block.create_var(name="Z", shape=[3], dtype='float32')
        op1 = block.append_op(
            type="abs", inputs={"X": [var1]}, outputs={"Out": [var2]})
        op2 = block.append_op(
            type="abs", inputs={"X": [var2]}, outputs={"Out": [var3]})
        res = amp.bf16.amp_utils.find_true_post_op(block.ops, op1, "Y")
        assert (res == [op2])

    def test_find_true_post_op_with_search_all(self):
        program = fluid.Program()
        block = program.current_block()
        startup_block = fluid.default_startup_program().global_block()

        var1 = block.create_var(name="X", shape=[3], dtype='float32')
        var2 = block.create_var(name="Y", shape=[3], dtype='float32')
        inititializer_op = startup_block._prepend_op(
            type="fill_constant",
            outputs={"Out": var1},
            attrs={"shape": var1.shape,
                   "dtype": var1.dtype,
                   "value": 1.0})

        op1 = block.append_op(
            type="abs", inputs={"X": [var1]}, outputs={"Out": [var2]})
        result = amp.bf16.amp_utils.find_true_post_op(
            block.ops, inititializer_op, "X", search_all=False)
        assert (len(result) == 0)
        result = amp.bf16.amp_utils.find_true_post_op(
            block.ops, inititializer_op, "X", search_all=True)
        assert (result == [op1])


if __name__ == '__main__':
    unittest.main()
