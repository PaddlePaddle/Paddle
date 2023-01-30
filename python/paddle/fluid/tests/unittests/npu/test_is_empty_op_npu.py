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
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import unittest
import numpy as np
import sys

sys.path.append("..")
from op_test import OpTest
import paddle

paddle.enable_static()


<<<<<<< HEAD
@unittest.skipIf(
    not paddle.is_compiled_with_npu(), "core is not compiled with NPU"
)
class TestEmpty(OpTest):
=======
@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestEmpty(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.op_type = "is_empty"
        self.set_data()

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def set_data(self):
        self.inputs = {'X': np.array([1, 2, 3]).astype(self.dtype)}
        self.outputs = {'Out': np.array([False])}

    def test_check_output(self):
        self.check_output_with_place(self.place)


<<<<<<< HEAD
@unittest.skipIf(
    not paddle.is_compiled_with_npu(), "core is not compiled with NPU"
)
class TestNotEmpty(TestEmpty):
=======
@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNotEmpty(TestEmpty):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_data(self):
        self.inputs = {'X': np.array([])}
        self.outputs = {'Out': np.array([True])}


<<<<<<< HEAD
@unittest.skipIf(
    not paddle.is_compiled_with_npu(), "core is not compiled with NPU"
)
class TestIsEmptyOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
=======
@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestIsEmptyOpError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            input_data = np.random.random((3, 2)).astype("float32")

            def test_Variable():
                # the input type must be Variable
                paddle.is_empty(x=input_data)

            self.assertRaises(TypeError, test_Variable)

            def test_type():
                # dtype must be float32, float16 in NPU
<<<<<<< HEAD
                x3 = paddle.static.data(
                    name="x3", shape=[4, 32, 32], dtype="bool"
                )
=======
                x3 = paddle.static.data(name="x3",
                                        shape=[4, 32, 32],
                                        dtype="bool")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                res = paddle.is_empty(x=x3)

            self.assertRaises(TypeError, test_type)

            def test_name_type():
                # name type must be string.
<<<<<<< HEAD
                x4 = paddle.static.data(
                    name="x4", shape=[3, 2], dtype="float32"
                )
=======
                x4 = paddle.static.data(name="x4",
                                        shape=[3, 2],
                                        dtype="float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                res = paddle.is_empty(x=x4, name=1)

            self.assertRaises(TypeError, test_name_type)


<<<<<<< HEAD
@unittest.skipIf(
    not paddle.is_compiled_with_npu(), "core is not compiled with NPU"
)
class TestIsEmptyOpDygraph(unittest.TestCase):
=======
@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestIsEmptyOpDygraph(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_dygraph(self):
        paddle.disable_static(paddle.NPUPlace(0))
        input = paddle.rand(shape=[4, 32, 32], dtype='float32')
        res = paddle.is_empty(x=input)


if __name__ == "__main__":
    unittest.main()
