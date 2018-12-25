# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from test_conv2d_transpose_op import TestConv2dTransposeOp, TestWithPad, TestWithStride


class TestMKLDNN(TestConv2dTransposeOp):
    def init_op_type(self):
        self.is_test = True
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.op_type = "conv2d_transpose"
        self._cpu_only = True

    def test_check_grad(self):
        return

    def test_check_grad_no_input(self):
        return

    def test_check_grad_no_filter(self):
        return


class TestMKLDNNWithPad(TestWithPad):
    def init_op_type(self):
        self.is_test = True
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.op_type = "conv2d_transpose"
        self._cpu_only = True

    def test_check_grad(self):
        return

    def test_check_grad_no_input(self):
        return

    def test_check_grad_no_filter(self):
        return


class TestMKLDNNWithStride(TestWithStride):
    def init_op_type(self):
        self.is_test = True
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.op_type = "conv2d_transpose"
        self._cpu_only = True

    def test_check_grad(self):
        return

    def test_check_grad_no_input(self):
        return

    def test_check_grad_no_filter(self):
        return


if __name__ == '__main__':
    unittest.main()
