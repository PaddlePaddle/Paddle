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

import paddle
from paddle.static import amp

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
        self.fp32_list.remove('tan')
        self.bf16_list.add('tan')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16({'tan'})

    def test_amp_lists_3(self):
        # 3. w={'lstm'}, b=None
        self.bf16_list.add('lstm')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16({'lstm'})

    def test_amp_lists_4(self):
        # 4. w=None, b={'matmul_v2'}
        self.bf16_list.remove('matmul_v2')
        self.fp32_list.add('matmul_v2')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_list={'matmul_v2'}
        )

    def test_amp_lists_5(self):
        # 5. w=None, b={'matmul_v2'}
        self.fp32_list.add('matmul_v2')
        self.bf16_list.remove('matmul_v2')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_list={'matmul_v2'}
        )

    def test_amp_lists_6(self):
        # 6. w=None, b={'lstm'}
        self.fp32_list.add('lstm')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_list={'lstm'}
        )

    def test_amp_lists_7(self):
        self.fp32_list.add('reshape2')
        self.gray_list.remove('reshape2')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_fp32_list={'reshape2'}
        )

    def test_amp_list_8(self):
        self.bf16_list.add('reshape2')
        self.gray_list.remove('reshape2')

        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(
            custom_bf16_list={'reshape2'}
        )


class AMPTest2(unittest.TestCase):
    def test_amp_lists_(self):
        # 7. w={'lstm'} b={'lstm'}
        # raise ValueError
        self.assertRaises(
            ValueError, amp.bf16.AutoMixedPrecisionListsBF16, {'lstm'}, {'lstm'}
        )


if __name__ == '__main__':
    unittest.main()
