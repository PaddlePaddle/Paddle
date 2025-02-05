#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

sys.path.append("../../legacy_test")
from test_conv3d_transpose_op import (
    TestConv3DTransposeOp,
    create_test_cudnn_bf16_class,
    create_test_cudnn_fp16_class,
)


class TestWithSymmetricPad_NHWC(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [2, 5, 5, 5, 3]  # NDHWC
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.data_format = 'NHWC'


class TestWithAsymmetricPad_NHWC(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 0, 1, 0, 1, 2]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [2, 5, 5, 5, 3]  # NDHWC
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.data_format = 'NHWC'


class TestWithGroups_NHWC(TestConv3DTransposeOp):
    def init_test_case(self):
        self.check_no_filter = True
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 2
        self.input_size = [2, 5, 5, 5, 4]  # NDHWC
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 3, 3, 3, 3]
        self.data_format = 'NHWC'


class TestWithStride_NHWC(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [2, 2, 2]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [2, 5, 5, 5, 3]  # NCDHW
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.data_format = 'NHWC'


class TestWithDilation_NHWC(TestConv3DTransposeOp):
    def init_test_case(self):
        self.check_no_input = True
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [2, 2, 2]
        self.groups = 1
        self.input_size = [2, 5, 5, 5, 3]  # NCDHW
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.data_format = 'NHWC'


# ----------------Conv3DTransposeCUDNN fp16----------------
create_test_cudnn_fp16_class(TestWithSymmetricPad_NHWC)
create_test_cudnn_fp16_class(TestWithAsymmetricPad_NHWC)
create_test_cudnn_fp16_class(TestWithGroups_NHWC)
create_test_cudnn_fp16_class(TestWithStride_NHWC)
create_test_cudnn_fp16_class(TestWithDilation_NHWC)


# ----------------Conv3DTransposeCUDNN bf16----------------
create_test_cudnn_bf16_class(TestWithSymmetricPad_NHWC)
create_test_cudnn_bf16_class(TestWithAsymmetricPad_NHWC)
create_test_cudnn_bf16_class(TestWithGroups_NHWC)
create_test_cudnn_bf16_class(TestWithStride_NHWC)
create_test_cudnn_bf16_class(TestWithDilation_NHWC)

if __name__ == '__main__':
    unittest.main()
