# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import sys
import unittest
=======
from __future__ import print_function
import unittest
import sys
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

sys.path.append("..")

import numpy as np
<<<<<<< HEAD
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
=======

import paddle
import paddle.fluid as fluid
from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class XPUTestUnsqueeze2Op(XPUOpTestWrapper):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = "unsqueeze2"
        self.use_dynamic_create_class = False

    class TestUnsqueeze2Op(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.op_type = "unsqueeze2"
            self.__class__.op_type = "unsqueeze2"
            self.use_mkldnn = False
            self.init_dtype()
            self.init_test_case()
            self.inputs = {
                "X": np.random.random(self.ori_shape).astype(self.dtype)
            }
            self.outputs = {
                "Out": self.inputs["X"].reshape(self.new_shape),
<<<<<<< HEAD
                "XShape": np.random.random(self.ori_shape).astype(self.dtype),
=======
                "XShape": np.random.random(self.ori_shape).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.init_attrs()

        def init_dtype(self):
            self.dtype = self.in_type

        def init_attrs(self):
            self.attrs = {"axes": self.axes}

        def init_test_case(self):
            self.ori_shape = (3, 40)
            self.axes = (1, 2)
            self.new_shape = (3, 1, 1, 40)

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, no_check_set=['XShape'])

        def test_check_grad(self):
            place = paddle.XPUPlace(0)
            if self.dtype == np.bool_:
                return
            else:
                self.check_grad_with_place(place, ['X'], 'Out')

    # Correct: Single input index.
    class TestUnsqueeze2Op1(TestUnsqueeze2Op):
<<<<<<< HEAD
        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (-1,)
=======

        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (-1, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.new_shape = (20, 5, 1)

    # Correct: Mixed input axis.
    class TestUnsqueeze2Op2(TestUnsqueeze2Op):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (0, -1)
            self.new_shape = (1, 20, 5, 1)

    # Correct: There is duplicated axis.
    class TestUnsqueeze2Op3(TestUnsqueeze2Op):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.ori_shape = (10, 2, 5)
            self.axes = (0, 3, 3)
            self.new_shape = (1, 10, 2, 1, 1, 5)

    # Correct: Reversed axes.
    class TestUnsqueeze2Op4(TestUnsqueeze2Op):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.ori_shape = (10, 2, 5)
            self.axes = (3, 1, 1)
            self.new_shape = (10, 1, 1, 2, 5, 1)

    # axes is a list(with tensor)
    class TestUnsqueeze2Op_AxesTensorList(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.op_type = "unsqueeze2"
            self.__class__.op_type = "unsqueeze2"
            self.use_mkldnn = False
            self.init_dtype()
            self.init_test_case()

            axes_tensor_list = []
            for index, ele in enumerate(self.axes):
<<<<<<< HEAD
                axes_tensor_list.append(
                    ("axes" + str(index), np.ones((1)).astype('int32') * ele)
                )

            self.inputs = {
                "X": np.random.random(self.ori_shape).astype(self.dtype),
                "AxesTensorList": axes_tensor_list,
=======
                axes_tensor_list.append(("axes" + str(index), np.ones(
                    (1)).astype('int32') * ele))

            self.inputs = {
                "X": np.random.random(self.ori_shape).astype(self.dtype),
                "AxesTensorList": axes_tensor_list
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.init_attrs()
            self.outputs = {
                "Out": self.inputs["X"].reshape(self.new_shape),
<<<<<<< HEAD
                "XShape": np.random.random(self.ori_shape).astype(self.dtype),
=======
                "XShape": np.random.random(self.ori_shape).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, no_check_set=['XShape'])

        def test_check_grad(self):
            place = paddle.XPUPlace(0)
            if self.dtype in [np.float32, np.float64, np.float16]:
                self.check_grad_with_place(place, ['X'], 'Out')
            else:
                return

        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (1, 2)
            self.new_shape = (20, 1, 1, 5)

        def init_attrs(self):
            self.attrs = {}

    class TestUnsqueeze2Op1_AxesTensorList(TestUnsqueeze2Op_AxesTensorList):
<<<<<<< HEAD
        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (-1,)
            self.new_shape = (20, 5, 1)

    class TestUnsqueeze2Op2_AxesTensorList(TestUnsqueeze2Op_AxesTensorList):
=======

        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (-1, )
            self.new_shape = (20, 5, 1)

    class TestUnsqueeze2Op2_AxesTensorList(TestUnsqueeze2Op_AxesTensorList):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (0, -1)
            self.new_shape = (1, 20, 5, 1)

    class TestUnsqueeze2Op3_AxesTensorList(TestUnsqueeze2Op_AxesTensorList):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.ori_shape = (10, 2, 5)
            self.axes = (0, 3, 3)
            self.new_shape = (1, 10, 2, 1, 1, 5)

    class TestUnsqueeze2Op4_AxesTensorList(TestUnsqueeze2Op_AxesTensorList):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.ori_shape = (10, 2, 5)
            self.axes = (3, 1, 1)
            self.new_shape = (10, 1, 1, 2, 5, 1)

    # axes is a Tensor
    class TestUnsqueeze2Op_AxesTensor(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.op_type = "unsqueeze2"
            self.__class__.op_type = "unsqueeze2"
            self.use_mkldnn = False
            self.init_test_case()
            self.init_dtype()

            self.inputs = {
                "X": np.random.random(self.ori_shape).astype(self.dtype),
<<<<<<< HEAD
                "AxesTensor": np.array(self.axes).astype("int32"),
=======
                "AxesTensor": np.array(self.axes).astype("int32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.init_attrs()
            self.outputs = {
                "Out": self.inputs["X"].reshape(self.new_shape),
<<<<<<< HEAD
                "XShape": np.random.random(self.ori_shape).astype(self.dtype),
=======
                "XShape": np.random.random(self.ori_shape).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, no_check_set=['XShape'])

        def test_check_grad(self):
            place = paddle.XPUPlace(0)
            if self.dtype in [np.float32, np.float64, np.float16]:
                self.check_grad_with_place(place, ['X'], 'Out')
            else:
                return

        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (1, 2)
            self.new_shape = (20, 1, 1, 5)

        def init_attrs(self):
            self.attrs = {}

    class TestUnsqueeze2Op1_AxesTensor(TestUnsqueeze2Op_AxesTensor):
<<<<<<< HEAD
        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (-1,)
            self.new_shape = (20, 5, 1)

    class TestUnsqueeze2Op2_AxesTensor(TestUnsqueeze2Op_AxesTensor):
=======

        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (-1, )
            self.new_shape = (20, 5, 1)

    class TestUnsqueeze2Op2_AxesTensor(TestUnsqueeze2Op_AxesTensor):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.ori_shape = (20, 5)
            self.axes = (0, -1)
            self.new_shape = (1, 20, 5, 1)

    class TestUnsqueeze2Op3_AxesTensor(TestUnsqueeze2Op_AxesTensor):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.ori_shape = (10, 2, 5)
            self.axes = (0, 3, 3)
            self.new_shape = (1, 10, 2, 1, 1, 5)

    class TestUnsqueeze2Op4_AxesTensor(TestUnsqueeze2Op_AxesTensor):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.ori_shape = (10, 2, 5)
            self.axes = (3, 1, 1)
            self.new_shape = (10, 1, 1, 2, 5, 1)


support_types = get_xpu_op_support_types("unsqueeze2")
for stype in support_types:
    create_test_class(globals(), XPUTestUnsqueeze2Op, stype)

if __name__ == "__main__":
    unittest.main()
