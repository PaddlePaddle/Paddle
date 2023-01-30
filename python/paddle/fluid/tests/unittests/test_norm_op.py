#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, skip_check_grad_ci

import paddle
import paddle.fluid as fluid
=======
from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from op_test import OpTest, skip_check_grad_ci
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def l2_norm(x, axis, epsilon):
    x2 = x**2
    s = np.sum(x2, axis=axis, keepdims=True)
    r = np.sqrt(s + epsilon)
    y = x / np.broadcast_to(r, x.shape)
    return y, r


class TestNormOp(OpTest):
<<<<<<< HEAD
    def setUp(self):
        self.op_type = "norm"
        self.python_api = paddle.nn.functional.normalize
=======

    def setUp(self):
        self.op_type = "norm"
        self.python_api = paddle.fluid.layers.l2_normalize
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.init_test_case()
        self.init_dtype()
        x = np.random.random(self.shape).astype(self.dtype)
        y, norm = l2_norm(x, self.axis, self.epsilon)
        self.inputs = {'X': x}
        self.attrs = {'epsilon': self.epsilon, 'axis': self.axis}
        self.outputs = {'Out': y, 'Norm': norm}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-8

    def init_dtype(self):
        self.dtype = "float64"


class TestNormOp2(TestNormOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.shape = [5, 3, 9, 7]
        self.axis = 0
        self.epsilon = 1e-8


class TestNormOp3(TestNormOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.shape = [5, 3, 2, 7]
        self.axis = -1
        self.epsilon = 1e-8


<<<<<<< HEAD
@skip_check_grad_ci(
    reason="'check_grad' on large inputs is too slow, "
    + "however it is desirable to cover the forward pass"
)
class TestNormOp4(TestNormOp):
=======
@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " +
                    "however it is desirable to cover the forward pass")
class TestNormOp4(TestNormOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.shape = [128, 1024, 14, 14]
        self.axis = 2
        self.epsilon = 1e-8

    def test_check_grad(self):
        pass


<<<<<<< HEAD
@skip_check_grad_ci(
    reason="'check_grad' on large inputs is too slow, "
    + "however it is desirable to cover the forward pass"
)
class TestNormOp5(TestNormOp):
=======
@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " +
                    "however it is desirable to cover the forward pass")
class TestNormOp5(TestNormOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.shape = [2048, 2048]
        self.axis = 1
        self.epsilon = 1e-8

    def test_check_grad(self):
        pass


class TestNormOp6(TestNormOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.008)


<<<<<<< HEAD
@unittest.skipIf(
    not fluid.core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestNormOp7(TestNormOp):
=======
@unittest.skipIf(not fluid.core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestNormOp7(TestNormOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = "float16"

    def test_check_output(self):
        self.check_output_with_place(fluid.core.CUDAPlace(0), atol=5e-2)

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            fluid.core.CUDAPlace(0), ['X'], 'Out', max_relative_error=0.05
        )
=======
        self.check_grad_with_place(fluid.core.CUDAPlace(0), ['X'],
                                   'Out',
                                   max_relative_error=0.05)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@skip_check_grad_ci(reason="skip check grad for test mode.")
class TestNormTestOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "norm"
        self.init_test_case()
        x = np.random.random(self.shape).astype("float64")
        y, norm = l2_norm(x, self.axis, self.epsilon)
        self.inputs = {'X': x}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
<<<<<<< HEAD
            'is_test': True,
=======
            'is_test': True
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': y}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-8


class API_NormTest(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_errors(self):
        with fluid.program_guard(fluid.Program()):

            def test_norm_x_type():
                data = fluid.data(name="x", shape=[3, 3], dtype="int64")
<<<<<<< HEAD
                out = paddle.nn.functional.normalize(data)
=======
                out = fluid.layers.l2_normalize(data)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_norm_x_type)


if __name__ == '__main__':
<<<<<<< HEAD
=======
    import paddle
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    paddle.enable_static()
    unittest.main()
