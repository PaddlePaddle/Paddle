#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
from utils import dygraph_guard, static_guard

import paddle


class TestSvdvalsOp(OpTest):
    def setUp(self):
        self.op_type = "svdvals"
        self.python_api = paddle.linalg.svdvals
        self.init_data()

    def init_data(self):
        """Generate input data and expected output."""
        self._input_shape = (100, 1)
        self._input_data = np.random.random(self._input_shape).astype("float64")
        self._output_data = np.linalg.svd(
            self._input_data, compute_uv=False, hermitian=False
        )
        self.inputs = {'x': self._input_data}
        self.outputs = {'s': self._output_data}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_svdvals_forward(self):
        """Check singular values calculation."""
        with dygraph_guard():
            dy_x = paddle.to_tensor(self._input_data)
            dy_s = paddle.linalg.svdvals(dy_x)
            np.testing.assert_allclose(
                dy_s.numpy(), self._output_data, rtol=1e-6, atol=1e-8
            )

    def test_check_grad(self):
        self.check_grad(['x'], ['s'], numeric_grad_delta=0.001, check_pir=True)


class TestSvdvalsBatched(TestSvdvalsOp):
    """Test svdvals operation with batched input."""

    def init_data(self):
        """Generate batched input matrix."""
        self._input_shape = (10, 3, 6)
        self._input_data = np.random.random(self._input_shape).astype("float64")

        self._output_data = np.linalg.svd(
            self._input_data, compute_uv=False, hermitian=False
        )

        self.inputs = {'x': self._input_data}
        self.outputs = {"s": self._output_data}


class TestSvdvalsBigMatrix(TestSvdvalsOp):
    def init_data(self):
        """Generate large input matrix."""
        self._input_shape = (40, 40)
        self._input_data = np.random.random(self._input_shape).astype("float64")
        self._output_data = np.linalg.svd(
            self._input_data, compute_uv=False, hermitian=False
        )
        self.inputs = {'x': self._input_data}
        self.outputs = {'s': self._output_data}

    def test_check_grad(self):
        self.check_grad(
            ['x'],
            ['s'],
            numeric_grad_delta=0.001,
            max_relative_error=1e-5,
            check_pir=True,
        )


class TestSvdvalsAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_dygraph_api(self):
        with dygraph_guard():
            x = paddle.to_tensor(self.x_np)
            # Test dynamic graph for svdvals
            s = paddle.linalg.svdvals(x)
            np_s = np.linalg.svd(self.x_np, compute_uv=False, hermitian=False)
            np.testing.assert_allclose(np_s, s.numpy(), rtol=1e-6)
            # Test with reshaped input
            x_reshaped = x.reshape([-1, 12, 10])
            s_reshaped = paddle.linalg.svdvals(x_reshaped)
            np_s_reshaped = np.array(
                [
                    np.linalg.svd(matrix, compute_uv=False, hermitian=False)
                    for matrix in self.x_np.reshape([-1, 12, 10])
                ]
            )
            np.testing.assert_allclose(
                np_s_reshaped, s_reshaped.numpy(), rtol=1e-6
            )

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data('x', [10, 12], dtype='float32')
                s = paddle.linalg.svdvals(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'x': self.x_np}, fetch_list=[s])

        np_s = np.linalg.svd(self.x_np, compute_uv=False, hermitian=False)
        for r in res:
            np.testing.assert_allclose(np_s, r, rtol=1e-6)

    def test_error(self):
        """Test invalid inputs for svdvals"""
        with paddle.base.dygraph.guard():

            def test_invalid_shape():
                """Test invalid shape input"""
                x_np_invalid_shape = np.random.uniform(-3, 3, [10]).astype(
                    'float32'
                )
                x_invalid_shape = paddle.to_tensor(x_np_invalid_shape)
                paddle.linalg.svdvals(x_invalid_shape)

            def test_empty_tensor():
                """Test empty tensor"""
                x_np_empty = np.empty([0, 10], dtype='float32')
                x_empty = paddle.to_tensor(x_np_empty)
                paddle.linalg.svdvals(x_empty)

            self.assertRaises(ValueError, test_invalid_shape)
            self.assertRaises(ValueError, test_empty_tensor)


if __name__ == "__main__":
    unittest.main()
