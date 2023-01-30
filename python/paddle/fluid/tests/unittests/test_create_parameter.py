# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
from paddle.fluid import Program, program_guard


class TestCreateParameterError(unittest.TestCase):
    def test_errors(self):
=======
import os, shutil
import unittest
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from paddle.fluid import ParamAttr, initializer
import paddle


class TestCreateParameterError(unittest.TestCase):

    def func_errors(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.enable_static()
        with program_guard(Program(), Program()):

            def test_shape():
<<<<<<< HEAD
                paddle.create_parameter(1, np.float32)
=======
                fluid.layers.create_parameter(1, np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_shape)

            def test_shape_item():
<<<<<<< HEAD
                paddle.create_parameter([1.0, 2.0, 3.0], "float32")
=======
                fluid.layers.create_parameter([1.0, 2.0, 3.0], "float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_shape_item)

            def test_attr():
<<<<<<< HEAD
                paddle.create_parameter(
                    [1, 2, 3], np.float32, attr=np.array([i for i in range(6)])
                )
=======
                fluid.layers.create_parameter([1, 2, 3],
                                              np.float32,
                                              attr=np.array(
                                                  [i for i in range(6)]))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_attr)

            def test_default_initializer():
<<<<<<< HEAD
                paddle.create_parameter(
                    [1, 2, 3],
                    np.float32,
                    default_initializer=np.array([i for i in range(6)]),
                )

            self.assertRaises(TypeError, test_default_initializer)

=======
                fluid.layers.create_parameter([1, 2, 3],
                                              np.float32,
                                              default_initializer=np.array(
                                                  [i for i in range(6)]))

            self.assertRaises(TypeError, test_default_initializer)

    def test_errors(self):
        with fluid.framework._test_eager_guard():
            self.func_errors()
        self.func_errors()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
