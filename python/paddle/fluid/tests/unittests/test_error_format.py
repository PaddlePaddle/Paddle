#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy
import os
import paddle
import paddle.compat as cpt
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import error_format
import unittest

_normal_traceback_str = "" \
    "File \"/work/self/startup.py\", line 15, in <module>\n" \
      "y_ = fluid.layers.fc(input=x, size=1, act=None)\n" \
    "PaddleEnfroceError: Expected x_mat_dims[1] == y_mat_dims[0], but received x_mat_dims[1]:13 != y_mat_dims[0]:12.\n" \
    "First matrix's width must be equal with second matrix's height. at [/work/paddle/paddle/fluid/operators/mul_op.cc:66]\n" \
      "[[{{operator mul}}]]"

_invalid_traceback_str = "" \
    "File \"/usr/local/lib/python3.5/dist-packages/paddle/fluid/layers/nn.py\", line 348, in fc\n" \
      "\"y_num_col_dims\": 1})\n" \
    "PaddleEnfroceError: Expected x_mat_dims[1] == y_mat_dims[0], but received x_mat_dims[1]:13 != y_mat_dims[0]:12.\n" \
    "First matrix's width must be equal with second matrix's height. at [/work/paddle/paddle/fluid/operators/mul_op.cc:66]\n" \
      "[[{{operator mul}}]]"


class TestErrorMessageHintAugment(unittest.TestCase):
    def augment_result_check(self, origin, augment_str):
        origin_augment = error_format.hint_augment(origin)
        origin_lines = origin.splitlines()
        origin_augment_lines = origin_augment.splitlines()
        origin_lines.reverse()
        origin_augment_lines.reverse()
        self.assertEqual(origin_augment_lines[0].strip(), augment_str)
        self.assertEqual("".join(origin_augment_lines[1:]),
                         "".join(origin_lines[1:]))

    def test_hint_augment_success(self):
        ex_msg = _normal_traceback_str
        augment_str = "[[operator mul execution error (defined at /work/self/startup.py:15)]]"
        self.augment_result_check(ex_msg, augment_str)

    def test_hint_augment_no_tag(self):
        ex_msg_lines = _normal_traceback_str.splitlines()
        # remove tag
        ex_msg_no_tag = "\n".join(ex_msg_lines[:-1])
        ex_msg_augment = error_format.hint_augment(ex_msg_no_tag)
        self.assertEqual(ex_msg_augment, ex_msg_no_tag)

    def test_hint_augment_no_frame(self):
        ex_msg_no_frame = _invalid_traceback_str
        augment_str = "[[operator mul error]]"
        self.augment_result_check(ex_msg_no_frame, augment_str)


if __name__ == "__main__":
    unittest.main()
