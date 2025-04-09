# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
np.random.seed(10)


class XPUTestMeshGridOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'meshgrid'
        self.use_dynamic_create_class = False

    class TestMeshGrid(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "meshgrid"
            self.place = paddle.XPUPlace(0)
            self.set_inputs()
            self.set_output()

        def init_dtype(self):
            self.dtype = self.in_type

        def init_test_data(self):
            self.shape = self.get_x_shape()
            ins = []
            outs = []
            for i in range(len(self.shape)):
                ins.append(
                    np.random.random((self.shape[i],)).astype(self.dtype)
                )

            for i in range(len(self.shape)):
                out_reshape = [1] * len(self.shape)
                out_reshape[i] = self.shape[i]
                out_temp = np.reshape(ins[i], out_reshape)
                outs.append(np.broadcast_to(out_temp, self.shape))
            return ins, outs

        def get_x_shape(self):
            return [100, 200]

        def set_inputs(self):
            ins, outs = self.init_test_data()
            self.inputs = {'X': [(f'x{i}', ins[i]) for i in range(len(ins))]}
            self.outputs = {
                'Out': [(f'out{i}', outs[i]) for i in range(len(outs))]
            }

        def set_output(self):
            pass

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

    class TestMeshgridOp2(TestMeshGrid):
        def get_x_shape(self):
            return [100, 300]


support_types = get_xpu_op_support_types('meshgrid')
for stype in support_types:
    create_test_class(globals(), XPUTestMeshGridOp, stype)

if __name__ == '__main__':
    unittest.main()
