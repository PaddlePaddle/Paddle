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

import os
import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci

import paddle
from paddle import base
from paddle.base import core


class TestSvdOp(OpTest):
    def setUp(self):
        paddle.enable_static()
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
        paddle.disable_static()
        dy_x = paddle.to_tensor(single_input)
        dy_u, dy_s, dy_vt = paddle.linalg.svd(dy_x)
        dy_out_x = dy_u.matmul(paddle.diag(dy_s)).matmul(dy_vt)
        if (paddle.abs(dy_out_x - dy_x) < 1e-7).all():
            ...
        else:
            print("EXPECTED:\n", dy_x)
            print("GOT     :\n", dy_out_x)
            raise RuntimeError("Check SVD Failed")
        paddle.enable_static()

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
        paddle.disable_static()
        a = np.random.rand(5, 5)
        x = paddle.to_tensor(a)
        u, s, vh = paddle.linalg.svd(x)
        gt_u, gt_s, gt_vh = np.linalg.svd(a, full_matrices=False)
        np.testing.assert_allclose(s, gt_s, rtol=1e-05)

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
                a = np.random.rand(5, 5)
                x = paddle.static.data(
                    name="input", shape=[5, 5], dtype='float64'
                )
                u, s, vh = paddle.linalg.svd(x)
                exe = paddle.static.Executor(place)
                gt_u, gt_s, gt_vh = np.linalg.svd(a, full_matrices=False)
                fetches = exe.run(
                    feed={"input": a},
                    fetch_list=[s],
                )
                np.testing.assert_allclose(fetches[0], gt_s, rtol=1e-05)

    def test_errors(self):
        with paddle.base.dygraph.guard():
            # The size of input in svd should not be 0.
            def test_0_size():
                array = np.array([], dtype=np.float32)
                x = paddle.to_tensor(np.reshape(array, [0, 0]), dtype='float32')
                paddle.linalg.svd(x, full_matrices=False)

            self.assertRaises(ValueError, test_0_size)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
