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

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestFillZerosLikeOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "fill_zeros_like"
        self.use_dynamic_create_class = False

    class TestXPUFillZerosLikeOp(XPUOpTest):
        def setUp(self):
            self.op_type = "fill_zeros_like"
            self.dtype = self.in_type

            self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
            self.outputs = {'Out': np.zeros_like(self.inputs['X'])}

        def init_dtype(self):
            pass

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))


support_types = get_xpu_op_support_types('fill_zeros_like')
for stype in support_types:
    create_test_class(globals(), XPUTestFillZerosLikeOp, stype)

if __name__ == "__main__":
    unittest.main()
