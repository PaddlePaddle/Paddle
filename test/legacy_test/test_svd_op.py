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

import itertools
import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci
from utils import dygraph_guard, static_guard

import paddle
from paddle import base
from paddle.base import core


class TestSvdOp(OpTest):
    def setUp(self):
        with static_guard():
            self.python_api = paddle.linalg.svd
            self.generate_input()
            self.generate_output()
            self.op_type = "svd"
            assert hasattr(self, "_output_data")
            self.inputs = {"X": self._input_data}
            self.attrs = {'full_matrices': self.get_full_matrices_option()}
            self.outputs = {
                "U": self._output_data[0],
                "S": self._output_data[1],
                "VH": self._output_data[2],
            }

    def _get_places(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        return places

    def generate_input(self):
        """return a input_data and input_shape"""
        self._input_shape = (100, 1)
        self._input_data = np.random.random(self._input_shape).astype("float64")

    def get_full_matrices_option(self):
        return False

    def generate_output(self):
        assert hasattr(self, "_input_data")
        self._output_data = np.linalg.svd(self._input_data)

    def test_check_output(self):
        self.check_output(no_check_set=['U', 'VH'], check_pir=True)

    def test_svd_forward(self):
        """u matmul diag(s) matmul vt must become X"""
        single_input = self._input_data.reshape(
            [-1, self._input_shape[-2], self._input_shape[-1]]
        )[0]
        with dygraph_guard():
            dy_x = paddle.to_tensor(single_input)
            dy_u, dy_s, dy_vt = paddle.linalg.svd(dy_x)
            dy_out_x = dy_u.matmul(paddle.diag(dy_s)).matmul(dy_vt)
            if (paddle.abs(dy_out_x - dy_x) < 1e-5).all():
                ...
            else:
                raise RuntimeError("Check SVD Failed")

    def check_S_grad(self):
        self.check_grad(['X'], ['S'], numeric_grad_delta=0.001, check_pir=True)

    def check_U_grad(self):
        self.check_grad(['X'], ['U'], numeric_grad_delta=0.001, check_pir=True)

    def check_V_grad(self):
        self.check_grad(['X'], ['VH'], numeric_grad_delta=0.001, check_pir=True)

    def test_check_grad(self):
        """
        remember the input matrix must be the full rank matrix, otherwise the gradient will stochatic because the u / v 's  (n-k) freedom  vectors
        """
        self.check_S_grad()
        self.check_U_grad()
        self.check_V_grad()


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestSvdOpComplexCase1(TestSvdOp):
    def generate_input(self):
        """return a input_data and input_shape"""
        self._input_shape = (5, 3)
        real_part = np.random.rand(*self._input_shape).astype("float32")
        imag_part = np.random.rand(*self._input_shape).astype("float32")
        self._input_data = real_part + 1j * imag_part

    def test_check_grad(self):
        places = self._get_places()
        with dygraph_guard():
            for place in places:
                x = paddle.to_tensor(
                    self._input_data, place=place, stop_gradient=False
                )
                U, s, Vh = paddle.linalg.svd(x, self.get_full_matrices_option())
                loss = (
                    paddle.sum(paddle.abs(U))
                    + paddle.sum(paddle.abs(s))
                    + paddle.sum(paddle.abs(Vh))
                )
                x_grad = paddle.grad(outputs=[loss], inputs=[x])


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestSvdOpComplexCase2(TestSvdOpComplexCase1):
    def generate_input(self):
        """return a input_data and input_shape"""
        self._input_shape = (3, 30)
        real_part = np.random.rand(*self._input_shape).astype("float32")
        imag_part = np.random.rand(*self._input_shape).astype("float32")
        self._input_data = real_part + 1j * imag_part


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestSvdOpComplexCase3(TestSvdOpComplexCase1):
    def generate_input(self):
        """return a input_data and input_shape"""
        self._input_shape = (100, 40)
        real_part = np.random.rand(*self._input_shape).astype("float64")
        imag_part = np.random.rand(*self._input_shape).astype("float64")
        self._input_data = real_part + 1j * imag_part


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestSvdOpComplexCase4(TestSvdOpComplexCase1):
    def generate_input(self):
        """return a input_data and input_shape"""
        self._input_shape = (100, 200)
        real_part = np.random.rand(*self._input_shape).astype("float64")
        imag_part = np.random.rand(*self._input_shape).astype("float64")
        self._input_data = real_part + 1j * imag_part

    def get_full_matrices_option(self):
        return True


class TestSvdCheckGrad2(TestSvdOp):
    # NOTE(xiongkun03): because we want to construct some full rank matrices,
    #                   so we can't specifize matrices which numel() > 100

    no_need_check_grad = True

    def generate_input(self):
        """return a deterministic  matrix, the range matrix;
        vander matrix must be a full rank matrix.
        """
        self._input_shape = (5, 5)
        self._input_data = (
            np.vander([2, 3, 4, 5, 6])
            .astype("float64")
            .reshape(self._input_shape)
        )


class TestSvdNormalMatrixSmall(TestSvdCheckGrad2):
    def generate_input(self):
        """small matrix SVD."""
        self._input_shape = (1, 1)
        self._input_data = np.random.random(self._input_shape).astype("float64")


class TestSvdNormalMatrix6x3(TestSvdCheckGrad2):
    def generate_input(self):
        """return a deterministic  matrix, the range matrix;
        vander matrix must be a full rank matrix.
        """
        self._input_shape = (6, 3)
        self._input_data = np.array(
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 5.0],
                [0.0, 0.0, 6.0],
                [2.0, 4.0, 9.0],
                [3.0, 6.0, 8.0],
                [3.0, 1.0, 0.0],
            ]
        ).astype("float64")


class TestSvdNormalMatrix3x6(TestSvdCheckGrad2):
    def generate_input(self):
        """return a deterministic  matrix, the range matrix;
        vander matrix must be a full rank matrix.
        """
        self._input_shape = (3, 6)
        self._input_data = np.array(
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 5.0],
                [0.0, 0.0, 6.0],
                [2.0, 4.0, 9.0],
                [3.0, 6.0, 8.0],
                [3.0, 1.0, 0.0],
            ]
        ).astype("float64")
        self._input_data = self._input_data.transpose((-1, -2))


class TestSvdNormalMatrix6x3Batched(TestSvdOp):
    def generate_input(self):
        self._input_shape = (10, 6, 3)
        self._input_data = np.array(
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 5.0],
                [0.0, 0.0, 6.0],
                [2.0, 4.0, 9.0],
                [3.0, 6.0, 8.0],
                [3.0, 1.0, 0.0],
            ]
        ).astype("float64")
        self._input_data = np.stack([self._input_data] * 10, axis=0)

    def test_svd_forward(self):
        """test_svd_forward not support batched input, so disable this test."""
        pass


class TestSvdNormalMatrix3x6Batched(TestSvdOp):
    def generate_input(self):
        """return a deterministic  matrix, the range matrix;
        vander matrix must be a full rank matrix.
        """
        self._input_shape = (10, 3, 6)
        self._input_data = np.array(
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 5.0],
                [0.0, 0.0, 6.0],
                [2.0, 4.0, 9.0],
                [3.0, 6.0, 8.0],
                [3.0, 1.0, 0.0],
            ]
        ).astype("float64")
        self._input_data = self._input_data.transpose((-1, -2))
        self._input_data = np.stack([self._input_data] * 10, axis=0)

    def test_svd_forward(self):
        """test_svd_forward not support batched input, so disable this test."""
        pass


class TestSvdNormalMatrix3x3x3x6Batched(TestSvdOp):
    def generate_input(self):
        """return a deterministic  matrix, the range matrix;
        vander matrix must be a full rank matrix.
        """
        self._input_shape = (3, 3, 3, 6)
        self._input_data = np.array(
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 5.0],
                [0.0, 0.0, 6.0],
                [2.0, 4.0, 9.0],
                [3.0, 6.0, 8.0],
                [3.0, 1.0, 0.0],
            ]
        ).astype("float64")
        self._input_data = self._input_data.transpose((-1, -2))
        self._input_data = np.stack(
            [self._input_data, self._input_data, self._input_data], axis=0
        )
        self._input_data = np.stack(
            [self._input_data, self._input_data, self._input_data], axis=0
        )

    def test_svd_forward(self):
        """test_svd_forward not support batched input, so disable this test."""
        pass


@skip_check_grad_ci(
    reason="'check_grad' on large inputs is too slow, "
    + "however it is desirable to cover the forward pass"
)
class TestSvdNormalMatrixBig(TestSvdOp):
    def generate_input(self):
        """big matrix SVD."""
        self._input_shape = (2, 200, 300)
        self._input_data = np.random.random(self._input_shape).astype("float64")

    def test_svd_forward(self):
        """test_svd_forward not support batched input, so disable this test."""
        pass

    def test_check_grad(self):
        pass


class TestSvdNormalMatrixBig2(TestSvdOp):
    def generate_input(self):
        """big matrix SVD."""
        self._input_shape = (1, 100)
        self._input_data = np.random.random(self._input_shape).astype("float64")


class TestSvdNormalMatrixFullMatrices(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_full_matrices(self):
        mat_shape = (2, 3)
        mat = np.random.random(mat_shape).astype("float64")
        x = paddle.to_tensor(mat)
        u, s, vh = paddle.linalg.svd(x, full_matrices=True)
        assert u.shape == [2, 2]
        assert vh.shape == [3, 3]
        x_recover = u.matmul(paddle.diag(s)).matmul(vh[0:2])
        if (paddle.abs(x_recover - x) > 1e-4).any():
            raise RuntimeError("mat can't be recovered\n")


class TestSvdFullMatriceGrad(TestSvdNormalMatrix6x3):
    def get_full_matrices_option(self):
        return True

    def test_svd_forward(self):
        """test_svd_forward not support full matrices, so disable this test."""
        pass

    def test_check_grad(self):
        """
        remember the input matrix must be the full rank matrix, otherwise the gradient will stochatic because the u / v 's  (n-k) freedom  vectors
        """
        self.check_S_grad()
        # self.check_U_grad() // don't check U grad, because U have freedom vector
        self.check_V_grad()


class TestSvdAPI(unittest.TestCase):
    def test_dygraph(self):
        def run_svd_dygraph(shape, dtype):
            if dtype == "float32":
                np_dtype = np.float32
            elif dtype == "float64":
                np_dtype = np.float64
            elif dtype == "complex64":
                np_dtype = np.complex64
            elif dtype == "complex128":
                np_dtype = np.complex128
            if np.issubdtype(np_dtype, np.complexfloating):
                a_dtype = np.float32 if np_dtype == np.complex64 else np.float64
                a_real = np.random.rand(*shape).astype(a_dtype)
                a_imag = np.random.rand(*shape).astype(a_dtype)
                a = a_real + 1j * a_imag
            else:
                a = np.random.rand(*shape).astype(np_dtype)

            places = []
            places.append(base.CPUPlace())
            if core.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                x = paddle.to_tensor(a, place=place)
                u, s, vh = paddle.linalg.svd(x)
                gt_u, gt_s, gt_vh = np.linalg.svd(a, full_matrices=False)
                np.testing.assert_allclose(s, gt_s, rtol=1e-05)

        with dygraph_guard():
            np.random.seed(7)
            tensor_shapes = [
                (0, 3),
                (3, 5),
                (5, 5),
                (5, 3),  # 2-dim Tensors
                (0, 3, 5),
                (4, 0, 5),
                (5, 4, 0),
                (4, 5, 3),  # 3-dim Tensors
                (0, 5, 3, 5),
                (2, 5, 3, 5),
                (3, 5, 5, 5),
                (4, 5, 5, 3),  # 4-dim Tensors
            ]
            dtypes = ["float32", "float64", 'complex64', 'complex128']
            for tensor_shape, dtype in itertools.product(tensor_shapes, dtypes):
                run_svd_dygraph(tensor_shape, dtype)

    def test_static(self):
        def run_svd_static(shape, dtype):
            if dtype == "float32":
                np_dtype = np.float32
            elif dtype == "float64":
                np_dtype = np.float64
            elif dtype == "complex64":
                np_dtype = np.complex64
            elif dtype == "complex128":
                np_dtype = np.complex128
            if np.issubdtype(np_dtype, np.complexfloating):
                a_dtype = np.float32 if np_dtype == np.complex64 else np.float64
                a_real = np.random.rand(*shape).astype(a_dtype)
                a_imag = np.random.rand(*shape).astype(a_dtype)
                a = a_real + 1j * a_imag
            else:
                a = np.random.rand(*shape).astype(np_dtype)

            places = []
            places.append(base.CPUPlace())
            if core.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x = paddle.static.data(
                        name="input", shape=shape, dtype=dtype
                    )
                    u, s, vh = paddle.linalg.svd(x)
                    exe = paddle.static.Executor(place)
                    gt_u, gt_s, gt_vh = np.linalg.svd(a, full_matrices=False)
                    fetches = exe.run(
                        feed={"input": a},
                        fetch_list=[s],
                    )
                    np.testing.assert_allclose(fetches[0], gt_s, rtol=1e-05)

            with static_guard():
                np.random.seed(7)
                tensor_shapes = [
                    (0, 3),
                    (3, 5),
                    (5, 5),
                    (5, 3),  # 2-dim Tensors
                    (0, 3, 5),
                    (4, 0, 5),
                    (5, 4, 0),
                    (4, 5, 3),  # 3-dim Tensors
                    (0, 5, 3, 5),
                    (2, 5, 3, 5),
                    (3, 5, 5, 5),
                    (4, 5, 5, 3),  # 4-dim Tensors
                ]
                dtypes = ["float32", "float64", 'complex64', 'complex128']
                for tensor_shape, dtype in itertools.product(
                    tensor_shapes, dtypes
                ):
                    run_svd_static(tensor_shape, dtype)


if __name__ == "__main__":
    unittest.main()
