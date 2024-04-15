#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class XPUTestArgsortOp_LargeN(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'argsort'
        self.use_dynamic_create_class = False

    class TestArgsortOpCase1(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "argsort"
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.axis = -1 if not hasattr(self, 'init_axis') else self.init_axis
            self.init_test_case()
            self.descending = (
                False
                if not hasattr(self, 'init_descending')
                else self.init_descending
            )

            np.random.seed(100)
            if self.dtype == np.float32:
                self.x = np.random.random(self.input_shape).astype(self.dtype)
            else:
                self.x = np.random.choice(
                    1000000, self.input_shape, replace=False
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

        def init_test_case(self):
            self.input_shape = [2, 8732]  # test for 8192 < n <= 10240

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, {'X'}, 'Out')

    class TestArgsortOpCase2(TestArgsortOpCase1):
        def init_test_case(self):
            self.input_shape = [2, 10241]  # test for 10240 < n <= 16384

    class TestArgsortOpCase3(TestArgsortOpCase1):
        def init_test_case(self):
            self.input_shape = [
                2,
                8732,
                1,
            ]  # test for 8192 < n <= 10240 + need_transpose
            self.axis = 1

    class TestArgsortOpCase4(TestArgsortOpCase1):
        def init_test_case(self):
            self.input_shape = [
                2,
                10241,
                1,
            ]  # test for 10240 < n <= 16384 + need_transpose
            self.axis = 1

    class TestStableArgsortOpCase1(XPUOpTest):
        def init_test_case(self):
            self.x = np.array([100.0, 50.0, 10.0] * 10)
            self.axis = -1
            self.descending = False

        def setUp(self):
            self.set_xpu()
            self.op_type = "argsort"
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.init_test_case()
            self.stable = True

            self.inputs = {"X": self.x}
            self.attrs = {
                "axis": self.axis,
                "descending": self.descending,
                "stable": self.stable,
            }
            self.get_output()
            self.outputs = {"Out": self.sorted_x, "Indices": self.indices}

        def get_output(self):
            if self.descending:
                self.indices = np.argsort(
                    -self.x, kind='stable', axis=self.axis
                )
                self.sorted_x = -np.sort(-self.x, kind='stable', axis=self.axis)
            else:
                self.indices = np.argsort(self.x, kind='stable', axis=self.axis)
                self.sorted_x = np.sort(self.x, kind='stable', axis=self.axis)

        def set_xpu(self):
            self.__class__.use_xpu = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, {'X'}, 'Out')

    class TestStableArgsortOpCase2(TestStableArgsortOpCase1):
        def init_test_case(self):
            self.x = np.array([100.0, 50.0, 10.0] * 10).reshape([30, 1])
            self.axis = 0
            self.descending = False

    class TestStableArgsortOpCase3(TestStableArgsortOpCase1):
        def init_test_case(self):
            self.x = np.array([100.0, 50.0, 10.0] * 10).reshape([1, 30])
            self.axis = 1
            self.descending = True

    class TestStableArgsortOpCase4(TestStableArgsortOpCase1):
        def init_test_case(self):
            self.x = np.array(
                [
                    [
                        [100.0, 50.0, -10.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [100.0, 50.0, -10.0, 1.0],
                    ],
                    [
                        [70.0, -30.0, 60.0, 100.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [100.0, 50.0, -10.0, 1.0],
                    ],
                ]
                * 20
            )
            self.axis = 0
            self.descending = True


support_types = get_xpu_op_support_types('argsort')
for stype in support_types:
    create_test_class(globals(), XPUTestArgsortOp, stype)
    if stype != "float16":
        # skip fp16 test on LARGE input because unstable sort on low-precision fp16 will lead to test failure
        create_test_class(globals(), XPUTestArgsortOp_LargeN, stype)

if __name__ == '__main__':
    unittest.main()
