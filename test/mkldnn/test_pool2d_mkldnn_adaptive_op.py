#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

sys.path.append("../legacy_test")

import numpy as np
from test_pool2d_op import (
    TestPool2D_Op,
    avg_pool2D_forward_naive,
)


def create_test_mkldnn_class(parent):
    class TestMKLDNNCase(parent):
        def init_kernel_type(self):
            self.use_mkldnn = True

        def init_data_type(self):
            self.dtype = np.float32

        def init_adaptive(self):
            self.adaptive = True

        def init_shape(self):
            self.shape = [1, 3, 8, 8]

    cls_name = "{}_{}".format(parent.__name__, "MKLDNNOp")
    TestMKLDNNCase.__name__ = cls_name
    globals()[cls_name] = TestMKLDNNCase


class TestAvgPoolAdaptive(TestPool2D_Op):
    def init_adaptive(self):
        self.adaptive = True

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_test_case(self):
        self.ksize = [1, 1]
        self.strides = [1, 1]

    def init_data_type(self):
        self.dtype = np.float32

    def init_global_pool(self):
        self.global_pool = False


class TestAsymPadCase1(TestAvgPoolAdaptive):
    def init_adaptive(self):
        self.adaptive = True

    def init_paddings(self):
        self.paddings = [0, 0, 0, 0]

    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_shape(self):
        self.shape = [1, 3, 8, 8]


create_test_mkldnn_class(TestPool2D_Op)
create_test_mkldnn_class(TestAvgPoolAdaptive)
create_test_mkldnn_class(TestAsymPadCase1)


if __name__ == '__main__':
    from paddle import enable_static

    enable_static()
    unittest.main()
