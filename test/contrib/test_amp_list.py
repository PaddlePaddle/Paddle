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
from paddle.static.amp.fp16_lists import AutoMixedPrecisionLists


class TestAMPList(unittest.TestCase):
    def setUp(self):
        self.default_black_list = [
            'linear_interp_v2',
            'nearest_interp_v2',
            'bilinear_interp_v2',
            'bicubic_interp_v2',
            'trilinear_interp_v2',
        ]
        self.custom_white_list = [
            'lookup_table',
            'lookup_table_v2',]
    
    def check_if_op_in_list(self, op_list, amp_list):
        for op in op_list:
            self.assertTrue(op in amp_list)

    def check_if_op_not_in_list(self, op_list, amp_list):
        for op in op_list:
            self.assertTrue(op not in amp_list)

    def test_static(self):
        amp_list = AutoMixedPrecisionLists(custom_white_list=self.custom_white_list)
        self.check_if_op_in_list(self.default_black_list, amp_list.black_list)
        self.check_if_op_in_list(self.custom_white_list, amp_list.white_list)
        self.check_if_op_not_in_list(self.custom_white_list, amp_list.black_list)
        self.check_if_op_not_in_list(self.custom_white_list, amp_list.unsupported_list)

    def test_eager(self):
        white_list = paddle.amp.white_list()
        black_list = paddle.amp.black_list()
        self.check_if_op_in_list(self.default_black_list, black_list["float16"]["O2"])
        self.check_if_op_not_in_list(['log', 'elementwise_add'], white_list)
        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}):
            out1 = paddle.rand([2, 3]) + paddle.rand([2, 3])
            out2 = paddle.log(out1)
        self.check_if_op_not_in_list(['log', 'elementwise_add'], white_list)
        self.assertEqual(out1.dtype, paddle.float16)
        self.assertEqual(out1.dtype, paddle.float32) 

if __name__ == "__main__":
    unittest.main()
