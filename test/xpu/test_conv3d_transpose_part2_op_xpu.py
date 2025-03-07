#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from test_conv3d_transpose_op_xpu import (
    XPUTestConv3DTransposeOp,
)


class XPUTestConv3DTransposePart2Op(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'conv3d_transpose'
        self.use_dynamic_create_class = False

    class TestWithSymmetricPad_NHWC(
        XPUTestConv3DTransposeOp.TestConv3DTransposeOp
    ):
        def init_test_case(self):
            self.pad = [1, 1, 1]
            self.stride = [1, 1, 1]
            self.dilations = [1, 1, 1]
            self.groups = 1
            self.input_size = [2, 5, 5, 5, 3]  # NDHWC
            f_c = self.input_size[-1]
            self.filter_size = [f_c, 6, 3, 3, 3]
            self.data_format = 'NHWC'

    class TestWithAsymmetricPad_NHWC(
        XPUTestConv3DTransposeOp.TestConv3DTransposeOp
    ):
        def init_test_case(self):
            self.pad = [1, 0, 1, 0, 1, 2]
            self.stride = [1, 1, 1]
            self.dilations = [1, 1, 1]
            self.groups = 1
            self.input_size = [2, 5, 5, 5, 3]  # NDHWC
            f_c = self.input_size[-1]
            self.filter_size = [f_c, 6, 3, 3, 3]
            self.data_format = 'NHWC'

    @unittest.skipIf(
        True,
        "XPU conv3_transpose ndhwc with groups not supported in xpudnn yet",
    )
    class TestWithGroups_NHWC(XPUTestConv3DTransposeOp.TestConv3DTransposeOp):
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

    class TestWithStride_NHWC(XPUTestConv3DTransposeOp.TestConv3DTransposeOp):
        def init_test_case(self):
            self.pad = [1, 1, 1]
            self.stride = [2, 2, 2]
            self.dilations = [1, 1, 1]
            self.groups = 1
            self.input_size = [2, 5, 5, 5, 3]  # NCDHW
            f_c = self.input_size[-1]
            self.filter_size = [f_c, 6, 3, 3, 3]
            self.data_format = 'NHWC'

    @unittest.skipIf(True, "dilation >= 2 not supported in xpudnn yet")
    class TestWithDilation_NHWC(XPUTestConv3DTransposeOp.TestConv3DTransposeOp):
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


support_types = get_xpu_op_support_types('conv3d_transpose')
for stype in support_types:
    create_test_class(globals(), XPUTestConv3DTransposePart2Op, stype)


if __name__ == '__main__':
    unittest.main()
