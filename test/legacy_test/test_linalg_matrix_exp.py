#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import unittest

import numpy as np
import scipy

import paddle
from paddle import base
from paddle.base import core

if sys.platform == 'win32':
    RTOL = {'float32': 1e-02, 'float64': 1e-04}
    ATOL = {'float32': 1e-02, 'float64': 1e-04}
else:
    RTOL = {'float32': 1e-06, 'float64': 1e-15}
    ATOL = {'float32': 1e-06, 'float64': 1e-15}


class MatrixExpTestCase(unittest.TestCase):
    def setUp(self):
        self.init_config()
        self.generate_input()
        self.generate_output()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def generate_input(self):
        self._input_shape = (5, 5)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype
        )

    def generate_output(self):
        self._output_data = scipy.linalg.expm(self._input_data)

    def init_config(self):
        self.dtype = 'float64'

    def test_dygraph(self):
        for place in self.places:
            paddle.disable_static(place)
            x = paddle.to_tensor(self._input_data, place=place)
            out = paddle.linalg.matrix_exp(x).numpy()

            np.testing.assert_allclose(
                out,
                self._output_data,
                rtol=RTOL.get(self.dtype),
                atol=ATOL.get(self.dtype),
            )

    # TODO(megemini): cond/while_loop should be tested in pir
    #
    def test_static(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

        for place in places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name="input",
                    shape=self._input_shape,
                    dtype=self._input_data.dtype,
                )

                out = paddle.linalg.matrix_exp(x)
                exe = paddle.static.Executor(place)

                res = exe.run(
                    feed={"input": self._input_data},
                    fetch_list=[out],
                )[0]

            np.testing.assert_allclose(
                res,
                self._output_data,
                rtol=RTOL.get(self.dtype),
                atol=ATOL.get(self.dtype),
            )

    def test_grad(self):
        for place in self.places:
            x = paddle.to_tensor(
                self._input_data, place=place, stop_gradient=False
            )
            out = paddle.linalg.matrix_exp(x)
            out.backward()
            x_grad = x.grad

            self.assertEqual(list(x_grad.shape), list(x.shape))
            self.assertEqual(x_grad.dtype, x.dtype)


class MatrixExpTestCaseFloat32(MatrixExpTestCase):
    def init_config(self):
        self.dtype = 'float32'


class MatrixExpTestCase3D(MatrixExpTestCase):
    def generate_input(self):
        self._input_shape = (2, 5, 5)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype
        )


class MatrixExpTestCase3DFloat32(MatrixExpTestCase3D):
    def init_config(self):
        self.dtype = 'float32'


class MatrixExpTestCase4D(MatrixExpTestCase):
    def generate_input(self):
        self._input_shape = (2, 3, 5, 5)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype
        )


class MatrixExpTestCase4DFloat32(MatrixExpTestCase4D):
    def init_config(self):
        self.dtype = 'float32'


class MatrixExpTestCaseEmpty(MatrixExpTestCase):
    def generate_input(self):
        self._input_shape = ()
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype
        )


class MatrixExpTestCaseEmptyFloat32(MatrixExpTestCaseEmpty):
    def init_config(self):
        self.dtype = 'float32'


class MatrixExpTestCaseScalar(MatrixExpTestCase):
    def generate_input(self):
        self._input_shape = (2, 3, 1, 1)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype
        )


class MatrixExpTestCaseScalarFloat32(MatrixExpTestCaseScalar):
    def init_config(self):
        self.dtype = 'float32'


# test precision for float32 with l1_norm comparing `conds`
class MatrixExpTestCasePrecisionFloat32L1norm0(MatrixExpTestCase):
    def init_config(self):
        self.dtype = 'float32'

    def generate_input(self):
        self._input_shape = (2, 2)
        self._input_data = np.array([[0, 0.2], [-0.2, 0]]).astype(self.dtype)


class MatrixExpTestCasePrecisionFloat32L1norm1(MatrixExpTestCase):
    def init_config(self):
        self.dtype = 'float32'

    def generate_input(self):
        self._input_shape = (2, 2)
        self._input_data = np.array([[0, 0.8], [-0.8, 0]]).astype(self.dtype)


class MatrixExpTestCasePrecisionFloat32L1norm2(MatrixExpTestCase):
    def init_config(self):
        self.dtype = 'float32'

    def generate_input(self):
        self._input_shape = (2, 2)
        self._input_data = np.array([[0, 2.0], [-2.0, 0]]).astype(self.dtype)


# test precision for float64 with l1_norm comparing `conds`
class MatrixExpTestCasePrecisionFloat64L1norm0(MatrixExpTestCase):
    def init_config(self):
        self.dtype = 'float64'

    def generate_input(self):
        self._input_shape = (2, 2)
        self._input_data = np.array([[0, 0.01], [-0.01, 0]]).astype(self.dtype)


class MatrixExpTestCasePrecisionFloat64L1norm1(MatrixExpTestCase):
    def init_config(self):
        self.dtype = 'float64'

    def generate_input(self):
        self._input_shape = (2, 2)
        self._input_data = np.array([[0, 0.1], [-0.1, 0]]).astype(self.dtype)


class MatrixExpTestCasePrecisionFloat64L1norm2(MatrixExpTestCase):
    def init_config(self):
        self.dtype = 'float64'

    def generate_input(self):
        self._input_shape = (2, 2)
        self._input_data = np.array([[0, 0.5], [-0.5, 0]]).astype(self.dtype)


class MatrixExpTestCasePrecisionFloat64L1norm3(MatrixExpTestCase):
    def init_config(self):
        self.dtype = 'float64'

    def generate_input(self):
        self._input_shape = (2, 2)
        self._input_data = np.array([[0, 1.5], [-1.5, 0]]).astype(self.dtype)


class MatrixExpTestCasePrecisionFloat64L1norm4(MatrixExpTestCase):
    def init_config(self):
        self.dtype = 'float64'

    def generate_input(self):
        self._input_shape = (2, 2)
        self._input_data = np.array([[0, 2.5], [-2.5, 0]]).astype(self.dtype)


# test error cases
class MatrixExpTestCaseError(unittest.TestCase):
    def test_error_dtype(self):
        with self.assertRaises(ValueError):
            x = np.array(123, dtype=int)
            paddle.linalg.matrix_exp(x)

    def test_error_ndim(self):
        # 1-d
        with self.assertRaises(ValueError):
            x = np.random.rand(1)
            paddle.linalg.matrix_exp(x)

        # not square
        with self.assertRaises(ValueError):
            x = np.random.rand(3, 4)
            paddle.linalg.matrix_exp(x)

        with self.assertRaises(ValueError):
            x = np.random.rand(2, 3, 4)
            paddle.linalg.matrix_exp(x)


if __name__ == '__main__':
    unittest.main()
