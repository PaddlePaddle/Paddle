#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

sys.path.append("..")
import sys
import unittest

import numpy as np

sys.path.append("..")
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
=======
from __future__ import print_function

import sys

sys.path.append("..")
import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class XPUTestFlattenOp(XPUOpTestWrapper):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'flatten_contiguous_range'
        self.use_dynamic_create_class = False

    class TestFlattenOp(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.set_xpu()
            self.op_type = "flatten_contiguous_range"
            self.place = paddle.XPUPlace(0)
            self.use_xpu = True
            self.use_mkldnn = False

            self.start_axis = 0
            self.stop_axis = -1
            self.dtype = self.in_type
            self.init_test_case()
            self.inputs = {
                "X": np.random.random(self.in_shape).astype(self.dtype)
            }
            self.init_attrs()
            self.outputs = {
                "Out": self.inputs["X"].reshape(self.new_shape),
<<<<<<< HEAD
                "XShape": np.random.random(self.in_shape).astype(self.dtype),
=======
                "XShape": np.random.random(self.in_shape).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

        def set_xpu(self):
            self.__class__.use_xpu = True

        def test_check_output(self):
            self.check_output_with_place(self.place, no_check_set=["XShape"])

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ["X"], "Out")

        def init_test_case(self):
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = -1
<<<<<<< HEAD
            self.new_shape = 120
=======
            self.new_shape = (120)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
                "stop_axis": self.stop_axis,
                'use_xpu': True,
            }

    class TestFlattenOp_1(TestFlattenOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 1
            self.stop_axis = 2
            self.new_shape = (3, 10, 4)

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
<<<<<<< HEAD
                "stop_axis": self.stop_axis,
            }

    class TestFlattenOp_2(TestFlattenOp):
=======
                "stop_axis": self.stop_axis
            }

    class TestFlattenOp_2(TestFlattenOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 1
            self.new_shape = (6, 5, 4)

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
<<<<<<< HEAD
                "stop_axis": self.stop_axis,
            }

    class TestFlattenOp_3(TestFlattenOp):
=======
                "stop_axis": self.stop_axis
            }

    class TestFlattenOp_3(TestFlattenOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 2
            self.new_shape = (30, 4)

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
<<<<<<< HEAD
                "stop_axis": self.stop_axis,
            }

    class TestFlattenOp_4(TestFlattenOp):
=======
                "stop_axis": self.stop_axis
            }

    class TestFlattenOp_4(TestFlattenOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = -2
            self.stop_axis = -1
            self.new_shape = (3, 2, 20)

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
<<<<<<< HEAD
                "stop_axis": self.stop_axis,
            }

    class TestFlattenOp_5(TestFlattenOp):
=======
                "stop_axis": self.stop_axis
            }

    class TestFlattenOp_5(TestFlattenOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 2
            self.stop_axis = 2
            self.new_shape = (3, 2, 5, 4)

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
<<<<<<< HEAD
                "stop_axis": self.stop_axis,
            }

    class TestFlattenOpSixDims(TestFlattenOp):
=======
                "stop_axis": self.stop_axis
            }

    class TestFlattenOpSixDims(TestFlattenOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.in_shape = (3, 2, 3, 2, 4, 4)
            self.start_axis = 3
            self.stop_axis = 5
            self.new_shape = (3, 2, 3, 32)

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
<<<<<<< HEAD
                "stop_axis": self.stop_axis,
            }

    class TestFlattenOp_Float32(TestFlattenOp):
=======
                "stop_axis": self.stop_axis
            }

    class TestFlattenOp_Float32(TestFlattenOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 1
            self.new_shape = (6, 5, 4)
            self.dtype = np.float32

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
<<<<<<< HEAD
                "stop_axis": self.stop_axis,
            }

    class TestFlattenOp_int32(TestFlattenOp):
=======
                "stop_axis": self.stop_axis
            }

    class TestFlattenOp_int32(TestFlattenOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 1
            self.new_shape = (6, 5, 4)
            self.dtype = np.int32

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
                "stop_axis": self.stop_axis,
<<<<<<< HEAD
                'use_xpu': True,
=======
                'use_xpu': True
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

        def test_check_grad(self):
            pass

    class TestFlattenOp_int8(TestFlattenOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 1
            self.new_shape = (6, 5, 4)
            self.dtype = np.int8

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
<<<<<<< HEAD
                "stop_axis": self.stop_axis,
=======
                "stop_axis": self.stop_axis
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

        def test_check_grad(self):
            pass

    class TestFlattenOp_int64(TestFlattenOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_test_case(self):
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 1
            self.new_shape = (6, 5, 4)
            self.dtype = np.int64

        def init_attrs(self):
            self.attrs = {
                "start_axis": self.start_axis,
<<<<<<< HEAD
                "stop_axis": self.stop_axis,
=======
                "stop_axis": self.stop_axis
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

        def test_check_grad(self):
            pass


class TestFlatten2OpError(unittest.TestCase):
<<<<<<< HEAD
    def test_errors(self):
        image_shape = (2, 3, 4, 4)
        x = (
            np.arange(
                image_shape[0]
                * image_shape[1]
                * image_shape[2]
                * image_shape[3]
            ).reshape(image_shape)
            / 100.0
        )
        x = x.astype('float32')

        def test_ValueError1():
            x_var = paddle.static.data(
                name="x", shape=image_shape, dtype='float32'
            )
=======

    def test_errors(self):
        image_shape = (2, 3, 4, 4)
        x = np.arange(image_shape[0] * image_shape[1] * image_shape[2] *
                      image_shape[3]).reshape(image_shape) / 100.
        x = x.astype('float32')

        def test_ValueError1():
            x_var = paddle.static.data(name="x",
                                       shape=image_shape,
                                       dtype='float32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out = paddle.flatten(x_var, start_axis=2, stop_axis=1)

        self.assertRaises(ValueError, test_ValueError1)

        def test_ValueError2():
<<<<<<< HEAD
            x_var = paddle.static.data(
                name="x", shape=image_shape, dtype='float32'
            )
=======
            x_var = paddle.static.data(name="x",
                                       shape=image_shape,
                                       dtype='float32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            paddle.flatten(x_var, start_axis=10, stop_axis=1)

        self.assertRaises(ValueError, test_ValueError2)

        def test_ValueError3():
<<<<<<< HEAD
            x_var = paddle.static.data(
                name="x", shape=image_shape, dtype='float32'
            )
=======
            x_var = paddle.static.data(name="x",
                                       shape=image_shape,
                                       dtype='float32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            paddle.flatten(x_var, start_axis=2, stop_axis=10)

        self.assertRaises(ValueError, test_ValueError3)

        def test_type():
            # dtype must be float32, float64, int8, int32, int64
<<<<<<< HEAD
            x2 = (
                np.arange(
                    image_shape[0]
                    * image_shape[1]
                    * image_shape[2]
                    * image_shape[3]
                ).reshape(image_shape)
                / 100.0
            )
            x2 = x2.astype('float16')
            x2_var = paddle.fluid.data(
                name='x2', shape=[3, 2, 4, 5], dtype='float16'
            )
=======
            x2 = np.arange(image_shape[0] * image_shape[1] * image_shape[2] *
                           image_shape[3]).reshape(image_shape) / 100.
            x2 = x2.astype('float16')
            x2_var = paddle.fluid.data(name='x2',
                                       shape=[3, 2, 4, 5],
                                       dtype='float16')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            paddle.flatten(x2_var)

        self.assertRaises(TypeError, test_type)

        def test_InputError():
            out = paddle.flatten(x)

        self.assertRaises(ValueError, test_InputError)


class TestStaticFlattenPythonAPI(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def execute_api(self, x, start_axis=0, stop_axis=-1):
        return paddle.flatten(x, start_axis, stop_axis)

    def test_static_api(self):
        paddle.enable_static()
        np_x = np.random.rand(2, 3, 4, 4).astype('float32')

        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
<<<<<<< HEAD
            x = paddle.static.data(
                name="x", shape=[2, 3, 4, 4], dtype='float32'
            )
=======
            x = paddle.static.data(name="x",
                                   shape=[2, 3, 4, 4],
                                   dtype='float32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out = self.execute_api(x, start_axis=-2, stop_axis=-1)

        exe = paddle.static.Executor(place=paddle.XPUPlace(0))
        fetch_out = exe.run(main_prog, feed={"x": np_x}, fetch_list=[out])
        self.assertTrue((2, 3, 16) == fetch_out[0].shape)


class TestStaticInplaceFlattenPythonAPI(TestStaticFlattenPythonAPI):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def execute_api(self, x, start_axis=0, stop_axis=-1):
        return x.flatten_(start_axis, stop_axis)


class TestFlattenPython(unittest.TestCase):
<<<<<<< HEAD
    def test_python_api(self):
        image_shape = (2, 3, 4, 4)
        x = (
            np.arange(
                image_shape[0]
                * image_shape[1]
                * image_shape[2]
                * image_shape[3]
            ).reshape(image_shape)
            / 100.0
        )
=======

    def test_python_api(self):
        image_shape = (2, 3, 4, 4)
        x = np.arange(image_shape[0] * image_shape[1] * image_shape[2] *
                      image_shape[3]).reshape(image_shape) / 100.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x = x.astype('float32')

        def test_InputError():
            out = paddle.flatten(x)

        self.assertRaises(ValueError, test_InputError)

        def test_Negative():
            paddle.disable_static(paddle.XPUPlace(0))
            img = paddle.to_tensor(x)
            out = paddle.flatten(img, start_axis=-2, stop_axis=-1)
            return out.numpy().shape

        res_shape = test_Negative()
        self.assertTrue((2, 3, 16) == res_shape)


support_types = get_xpu_op_support_types('flatten_contiguous_range')
for stype in support_types:
    create_test_class(globals(), XPUTestFlattenOp, stype)

if __name__ == "__main__":
    unittest.main()
