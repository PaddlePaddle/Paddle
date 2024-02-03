#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class XPUTestReduceAmaxOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'reduce_amin'

    class XPUTestReduceAmaxBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.set_case()

        def set_case(self):
            self.op_type = 'reduce_amin'
            self.shape = (20, 10)
            self.attrs = {'use_xpu': True, 'keep_dim': False, 'dim': (1,)}

            self.inputs = {
                'X': np.random.randint(0, 100, self.shape).astype("float32")
            }

            expect_input = self.inputs['X']
            self.outputs = {
                'Out': np.amin(
                    expect_input,
                    axis=self.attrs['dim'],
                    keepdims=self.attrs['keep_dim'],
                )
            }

        def test_check_output(self):
            self.check_output_with_place(self.place)


support_types = get_xpu_op_support_types('reduce_amin')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceAmaxOp, stype)

if __name__ == '__main__':
    unittest.main()
