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
import sys

sys.path.append("..")
import paddle
import paddle.fluid.core as core
from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestOneHotOP(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'one_hot'
        self.use_dynamic_create_class = False

    class TestXPUOneHotOP(XPUOpTest):

        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.op_type = 'one_hot'

            self.set_data()
            self.set_input()

        def set_data(self):
            self.depth = 10
            self.depth_np = np.array(10).astype('int32')
            self.x_lod = [[4, 1, 3, 3]]
            self.x = [
                np.random.randint(0, self.depth - 1)
                for i in range(sum(self.x_lod[0]))
            ]
            self.x = np.array(self.x).astype(self.dtype).reshape(
                [sum(self.x_lod[0]), 1])

            self.out = np.zeros(shape=(np.product(self.x.shape[:-1]),
                                       self.depth)).astype('float32')
            for i in range(np.product(self.x.shape)):
                self.out[i, self.x[i]] = 1.0

            self.outputs = {'Out': (self.out, self.x_lod)}

        def set_input(self):
            self.inputs = {
                'X': (self.x, self.x_lod),
                'depth_tensor': self.depth_np
            }
            self.attrs = {'dtype': int(core.VarDesc.VarType.FP32)}

        def test_check_output(self):
            self.check_output(check_dygraph=False)

        def init_dtype(self):
            self.dtype = self.in_type

    class TestXPUOneHotOP_attr(TestXPUOneHotOP):

        def set_input(self):
            self.inputs = {'X': (self.x, self.x_lod)}
            self.attrs = {
                'dtype': int(core.VarDesc.VarType.FP32),
                'depth': self.depth
            }

    class TestXPUOneHotOP_default_dtype(TestXPUOneHotOP):

        def set_input(self):
            self.inputs = {
                'X': (self.x, self.x_lod),
                'depth_tensor': self.depth_np
            }
            self.attrs = {}

    class TestXPUOneHotOP_default_dtype_attr(TestXPUOneHotOP):

        def set_input(self):
            self.inputs = {'X': (self.x, self.x_lod)}
            self.attrs = {'depth': self.depth}

    class TestXPUOneHotOP_out_of_range(TestXPUOneHotOP):

        def set_data(self):
            self.depth = 10
            self.x_lod = [[4, 1, 3, 3]]
            self.x = [
                np.random.choice([-1, self.depth])
                for i in range(sum(self.x_lod[0]))
            ]
            self.x = np.array(self.x).astype(self.dtype).reshape(
                [sum(self.x_lod[0]), 1])

            self.out = np.zeros(shape=(np.product(self.x.shape[:-1]),
                                       self.depth)).astype('float32')

            self.outputs = {'Out': (self.out, self.x_lod)}

        def set_input(self):
            self.inputs = {'X': (self.x, self.x_lod)}
            self.attrs = {'depth': self.depth, 'allow_out_of_range': True}


support_types = get_xpu_op_support_types('one_hot')
for stype in support_types:
    create_test_class(globals(), XPUTestOneHotOP, stype)

if __name__ == "__main__":
    unittest.main()
