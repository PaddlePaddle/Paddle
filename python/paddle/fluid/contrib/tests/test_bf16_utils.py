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
from paddle.fluid import core
from paddle.fluid.contrib.mixed_precision.bf16 import amp_utils
from paddle.fluid.contrib.mixed_precision.bf16 import amp_lists
from paddle.fluid.contrib.mixed_precision.bf16 import AutoMixedPrecisionLists
import paddle

paddle.enable_static()


class AMPTest(unittest.TestCase):
    def test_amp_lists(self):
        bf16_list = copy.copy(amp_lists.bf16_list)
        fp32_list = copy.copy(amp_lists.fp32_list)
        gray_list = copy.copy(amp_lists.gray_list)

        amp_lists_ = AutoMixedPrecisionLists()
        self.assertEqual(amp_lists_.bf16_list, bf16_list)
        self.assertEqual(amp_lists_.fp32_list, fp32_list)
        self.assertEqual(amp_lists_.gray_list, gray_list)

    def test_amp_lists_1(self):
        bf16_list = copy.copy(amp_lists.bf16_list)
        fp32_list = copy.copy(amp_lists.fp32_list)
        gray_list = copy.copy(amp_lists.gray_list)

        # 1. w={'exp}, b=None
        bf16_list.add('exp')
        fp32_list.remove('exp')

        amp_lists_ = AutoMixedPrecisionLists({'exp'})
        self.assertEqual(amp_lists_.bf16_list, bf16_list)
        self.assertEqual(amp_lists_.fp32_list, fp32_list)
        self.assertEqual(amp_lists_.gray_list, gray_list)

    def test_amp_lists_2(self):
        bf16_list = copy.copy(amp_lists.bf16_list)
        fp32_list = copy.copy(amp_lists.fp32_list)
        gray_list = copy.copy(amp_lists.gray_list)

        # 2. w={'tanh'}, b=None
        fp32_list.remove('tanh')
        bf16_list.add('tanh')

        amp_lists_ = AutoMixedPrecisionLists({'tanh'})
        self.assertEqual(amp_lists_.bf16_list, bf16_list)
        self.assertEqual(amp_lists_.fp32_list, fp32_list)
        self.assertEqual(amp_lists_.gray_list, gray_list)

    def test_amp_lists_3(self):
        bf16_list = copy.copy(amp_lists.bf16_list)
        fp32_list = copy.copy(amp_lists.fp32_list)
        gray_list = copy.copy(amp_lists.gray_list)

        # 3. w={'lstm'}, b=None
        bf16_list.add('lstm')

        amp_lists_ = AutoMixedPrecisionLists({'lstm'})
        self.assertEqual(amp_lists_.bf16_list, bf16_list)
        self.assertEqual(amp_lists_.fp32_list, fp32_list)
        self.assertEqual(amp_lists_.gray_list, gray_list)

    def test_amp_lists_4(self):
        bf16_list = copy.copy(amp_lists.bf16_list)
        fp32_list = copy.copy(amp_lists.fp32_list)
        gray_list = copy.copy(amp_lists.gray_list)

        # 4. w=None, b={'elementwise_add'}
        bf16_list.remove('elementwise_add')
        fp32_list.add('elementwise_add')

        amp_lists_ = AutoMixedPrecisionLists(
            custom_fp32_list={'elementwise_add'})
        self.assertEqual(amp_lists_.bf16_list, bf16_list)
        self.assertEqual(amp_lists_.fp32_list, fp32_list)
        self.assertEqual(amp_lists_.gray_list, gray_list)

    def test_amp_lists_5(self):
        bf16_list = copy.copy(amp_lists.bf16_list)
        fp32_list = copy.copy(amp_lists.fp32_list)
        gray_list = copy.copy(amp_lists.gray_list)

        # 5. w=None, b={'elementwise_add'}
        fp32_list.add('elementwise_add')
        bf16_list.remove('elementwise_add')

        amp_lists_ = AutoMixedPrecisionLists(
            custom_fp32_list={'elementwise_add'})
        self.assertEqual(amp_lists_.bf16_list, bf16_list)
        self.assertEqual(amp_lists_.fp32_list, fp32_list)
        self.assertEqual(amp_lists_.gray_list, gray_list)

    def test_amp_lists_6(self):
        bf16_list = copy.copy(amp_lists.bf16_list)
        fp32_list = copy.copy(amp_lists.fp32_list)
        gray_list = copy.copy(amp_lists.gray_list)

        # 6. w=None, b={'lstm'}
        fp32_list.add('lstm')

        amp_lists_ = AutoMixedPrecisionLists(custom_fp32_list={'lstm'})
        self.assertEqual(amp_lists_.bf16_list, bf16_list)
        self.assertEqual(amp_lists_.fp32_list, fp32_list)
        self.assertEqual(amp_lists_.gray_list, gray_list)

    def test_amp_lists_7(self):
        # 7. w={'lstm'} b={'lstm'}
        # raise ValueError
        self.assertRaises(ValueError, AutoMixedPrecisionLists, {'lstm'},
                          {'lstm'})

    def test_find_op_index(self):
        block = fluid.default_main_program().global_block()
        op_desc = core.OpDesc()
        idx = amp_utils.find_op_index(block.desc, op_desc)
        assert (idx == -1)

    def test_find_true_post_op(self):
        block = fluid.default_main_program().global_block()

        var1 = block.create_var(name="X", shape=[3], dtype='float32')
        var2 = block.create_var(name="Y", shape=[3], dtype='float32')
        var3 = block.create_var(name="Z", shape=[3], dtype='float32')
        op1 = block.append_op(
            type="abs", inputs={"X": [var1]}, outputs={"Out": [var2]})
        op2 = block.append_op(
            type="abs", inputs={"X": [var2]}, outputs={"Out": [var3]})
        res = amp_utils.find_true_post_op(block.ops, op1, "Y")
        assert (res == [op2])


if __name__ == '__main__':
    unittest.main()
