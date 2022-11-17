#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import unittest
import sys

sys.path.append("..")

import paddle
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    create_test_class,
    get_xpu_op_support_types,
    XPUOpTestWrapper,
)

paddle.enable_static()


class XPUTestArgsortOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'argsort'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        base_class = self.TestArgsortOp
        classes = []
        for descending in [True, False]:
            for axis in [0, 1, 2, -1, -2]:
                class_name = (
                    'XPUTestArgsortOp_axis_' + str(axis) + '_' + str(descending)
                )
                attr_dict = {'init_axis': axis, 'init_descending': descending}
                classes.append([class_name, attr_dict])
        return base_class, classes

    class TestArgsortOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "argsort"
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.input_shape = (2, 2, 2, 3, 3)
            self.axis = -1 if not hasattr(self, 'init_axis') else self.init_axis
            self.descending = (
                False
                if not hasattr(self, 'init_descending')
                else self.init_descending
            )

            if self.dtype == np.float32:
                self.x = np.random.random(self.input_shape).astype(self.dtype)
            else:
                self.x = np.random.randint(
                    low=-1000, high=1000, size=self.input_shape
                ).astype(self.dtype)

            self.inputs = {"X": self.x}
            self.attrs = {"axis": self.axis, "descending": self.descending}
            self.get_output()
            self.outputs = {"Out": self.sorted_x, "Indices": self.indices}

        def get_output(self):
            if self.descending:
                self.indices = np.flip(
                    np.argsort(self.x, kind='heapsort', axis=self.axis),
                    self.axis,
                )
                self.sorted_x = np.flip(
                    np.sort(self.x, kind='heapsort', axis=self.axis), self.axis
                )
            else:
                self.indices = np.argsort(
                    self.x, kind='heapsort', axis=self.axis
                )
                self.sorted_x = np.sort(self.x, kind='heapsort', axis=self.axis)

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, {'X'}, 'Out')


support_types = get_xpu_op_support_types('argsort')
for stype in support_types:
    create_test_class(globals(), XPUTestArgsortOp, stype)

if __name__ == '__main__':
    unittest.main()
