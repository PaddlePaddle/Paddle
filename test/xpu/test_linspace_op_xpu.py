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


class XPUTestLinspaceOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'linspace'
        self.use_dynamic_create_class = False

    class TestXPULinespaceOp(XPUOpTest):
        def setUp(self):
            self.op_type = "linspace"
            self.dtype = self.in_type
            self.set_attrs()

            self.atol = 1e-4
            np.random.seed(10)
            self.inputs = {
                'Start': np.array([0]).astype(self.dtype),
                'Stop': np.array([10]).astype(self.dtype),
                'Num': np.array([11]).astype('int32'),
            }
            self.outputs = {'Out': np.arange(0, 11).astype(self.dtype)}
            self.attrs = {
                'dtype': int(
                    paddle.base.framework.convert_np_dtype_to_proto_type(
                        self.dtype
                    )
                )
            }

        def set_attrs(self):
            pass

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0), atol=self.atol)

    class TestXPULinespace2(TestXPULinespaceOp):
        def set_attrs(self):
            self.inputs = {
                'Start': np.array([10]).astype(self.dtype),
                'Stop': np.array([0]).astype(self.dtype),
                'Num': np.array([11]).astype('int32'),
            }

            self.outputs = {'Out': np.arange(10, -1, -1).astype(self.dtype)}

    class TestXPULinespace3(TestXPULinespaceOp):
        def set_attrs(self):
            self.inputs = {
                'Start': np.array([10]).astype(self.dtype),
                'Stop': np.array([0]).astype(self.dtype),
                'Num': np.array([1]).astype('int32'),
            }
            self.outputs = {'Out': np.array(10, dtype=self.dtype)}

    class TestXPULinespace4(TestXPULinespaceOp):
        def set_attrs(self):
            self.inputs = {
                'Start': np.array([0]).astype(self.dtype),
                'Stop': np.array([10]).astype(self.dtype),
                'Num': np.array([0]).astype('int32'),
            }
            self.outputs = {'Out': np.array([], dtype=self.dtype)}


support_types = get_xpu_op_support_types('linspace')
for stype in support_types:
    create_test_class(globals(), XPUTestLinspaceOp, stype)


if __name__ == "__main__":
    unittest.main()
