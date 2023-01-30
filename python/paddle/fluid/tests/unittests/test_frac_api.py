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

=======
from __future__ import print_function

import unittest
import numpy as np
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
<<<<<<< HEAD
=======
from paddle.fluid.framework import _test_eager_guard
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def ref_frac(x):
    return x - np.trunc(x)


class TestFracAPI(unittest.TestCase):
    """Test Frac API"""

    def set_dtype(self):
        self.dtype = 'float64'

    def setUp(self):
        self.set_dtype()
        self.x_np = np.random.uniform(-3, 3, [2, 3]).astype(self.dtype)
<<<<<<< HEAD
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
=======
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_api_static(self):
        paddle.enable_static()
        with program_guard(Program()):
            input = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = paddle.frac(input)
            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
<<<<<<< HEAD
            (res,) = exe.run(feed={'X': self.x_np}, fetch_list=[out])
=======
            res, = exe.run(feed={'X': self.x_np}, fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        out_ref = ref_frac(self.x_np)
        np.testing.assert_allclose(out_ref, res, rtol=1e-05)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = paddle.frac(x)
        out_ref = ref_frac(self.x_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

    def test_api_eager(self):
        paddle.disable_static(self.place)
<<<<<<< HEAD
        x_tensor = paddle.to_tensor(self.x_np)
        out = paddle.frac(x_tensor)
=======
        with _test_eager_guard():
            x_tensor = paddle.to_tensor(self.x_np)
            out = paddle.frac(x_tensor)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        out_ref = ref_frac(self.x_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()

<<<<<<< HEAD
=======
    def test_api_eager_dygraph(self):
        with _test_eager_guard():
            self.test_api_dygraph()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

class TestFracInt32(TestFracAPI):
    """Test Frac API with data type int32"""

    def set_dtype(self):
        self.dtype = 'int32'


class TestFracInt64(TestFracAPI):
    """Test Frac API with data type int64"""

    def set_dtype(self):
        self.dtype = 'int64'


class TestFracFloat32(TestFracAPI):
    """Test Frac API with data type float32"""

    def set_dtype(self):
        self.dtype = 'float32'


class TestFracError(unittest.TestCase):
    """Test Frac Error"""

    def setUp(self):
        self.x_np = np.random.uniform(-3, 3, [2, 3]).astype('int16')
<<<<<<< HEAD
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
=======
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_static_error(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [5, 5], 'bool')
            self.assertRaises(TypeError, paddle.frac, x)

    def test_dygraph_error(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np, dtype='int16')
        self.assertRaises(TypeError, paddle.frac, x)


if __name__ == '__main__':
    unittest.main()
