# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, get_places
from utils import dygraph_guard, static_guard

import paddle
from paddle import base, static
from paddle.base import core

paddle.enable_static()
SEED = 2049
np.random.seed(SEED)


def matrix_rank_wrapper(x, tol=None, use_default_tol=True, hermitian=False):
    return paddle.linalg.matrix_rank(x, tol, hermitian)


class TestMatrixRankOP(OpTest):
    def setUp(self):
        self.python_api = matrix_rank_wrapper
        self.op_type = "matrix_rank"
        self.init_data()
        target_dtype = (
            np.float32
            if self.x.dtype == np.complex64
            else (np.float64 if self.x.dtype == np.complex128 else self.x.dtype)
        )
        self.inputs = {'X': self.x}
        self.attrs = {'hermitian': self.hermitian}
        if self.tol_tensor is not None:
            if self.tol_tensor.dtype != target_dtype:
                self.inputs['TolTensor'] = self.tol_tensor.astype(target_dtype)
            else:
                self.inputs["TolTensor"] = self.tol_tensor
        if self.tol is not None:
            self.attrs["tol"] = self.tol
        self.attrs["use_default_tol"] = self.use_default_tol
        self.outputs = {'Out': self.out}

    def _get_places(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        return places

    def test_check_output(self):
        self.check_output(check_pir=True)

    def init_data(self):
        self.x = np.eye(3, dtype=np.float32)
        self.tol_tensor = None
        self.tol = 0.1
        self.use_default_tol = False
        self.hermitian = True
        self.out = np.linalg.matrix_rank(self.x, self.tol, self.hermitian)


class TestMatrixRankOP1(TestMatrixRankOP):
    def init_data(self):
        self.x = np.eye(3, k=1, dtype=np.float64)
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = True
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


class TestMatrixRankOP2(TestMatrixRankOP):
    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.tol_tensor = np.random.random([3, 4]).astype(self.x.dtype)
        self.tol = None
        self.use_default_tol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


class TestMatrixRankOP3(TestMatrixRankOP):
    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = True
        self.hermitian = True
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


class TestMatrixRankOP4(TestMatrixRankOP):
    def init_data(self):
        self.x = np.random.rand(1, 10).astype(np.float32)
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = True
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


class TestMatrixRankOP5(TestMatrixRankOP):
    def init_data(self):
        self.x = np.random.rand(5, 1).astype(np.float64)
        self.tol_tensor = np.random.random([1, 4]).astype(self.x.dtype)
        self.tol = None
        self.use_default_tol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


class TestMatrixRankOP6(TestMatrixRankOP):
    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


class TestMatrixRankOP7(TestMatrixRankOP):
    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.tol_tensor = np.random.random([200, 200]).astype(self.x.dtype)
        self.tol = None
        self.use_default_tol = True
        self.hermitian = True
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestMatrixRankComplexOP(TestMatrixRankOP):
    def init_data(self):
        x_real = np.eye(3, dtype=np.float32)
        x_imag = np.eye(3, dtype=np.float32)
        self.x = x_real + 1j * x_imag
        self.tol_tensor = None
        self.tol = 0.1
        self.use_default_tol = False
        self.hermitian = True
        self.out = np.linalg.matrix_rank(self.x, self.tol, self.hermitian)


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestMatrixRankComplexOP1(TestMatrixRankOP):
    def init_data(self):
        x_real = np.eye(3, k=1, dtype=np.float64)
        x_imag = np.eye(3, k=1, dtype=np.float64)
        self.x = x_real + 1j * x_imag
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = True
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestMatrixRankComplexOP2(TestMatrixRankOP):
    def init_data(self):
        x_real = np.random.rand(3, 4, 5, 6).astype(np.float32)
        x_imag = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.x = x_real + 1j * x_imag
        self.tol_tensor = np.random.random([3, 4]).astype(x_real.dtype)
        self.tol = None
        self.use_default_tol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestMatrixRankComplexOP3(TestMatrixRankOP):
    def init_data(self):
        x_real = np.eye(200, dtype=np.float64)
        x_imag = np.eye(200, dtype=np.float64)
        self.x = x_real + 1j * x_imag
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = True
        self.hermitian = True
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestMatrixRankComplexOP4(TestMatrixRankOP):
    def init_data(self):
        x_real = np.random.rand(1, 10).astype(np.float32)
        x_imag = np.random.rand(1, 10).astype(np.float32)
        self.x = x_real + 1j * x_imag
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = True
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestMatrixRankComplexOP5(TestMatrixRankOP):
    def init_data(self):
        x_real = np.random.rand(5, 1).astype(np.float64)
        x_imag = np.random.rand(5, 1).astype(np.float64)
        self.x = x_real + 1j * x_imag
        self.tol_tensor = np.random.random([1, 4]).astype(x_real.dtype)
        self.tol = None
        self.use_default_tol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestMatrixRankComplexOP6(TestMatrixRankOP):
    def init_data(self):
        x_real = np.random.rand(3, 4, 5, 6).astype(np.float32)
        x_imag = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.x = x_real + 1j * x_imag
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestMatrixRankComplexOP7(TestMatrixRankOP):
    def init_data(self):
        x_real = np.eye(200, dtype=np.float64)
        x_imag = np.eye(200, dtype=np.float64)
        self.x = x_real + 1j * x_imag
        self.tol_tensor = np.random.random([200, 200]).astype(x_real.dtype)
        self.tol = None
        self.use_default_tol = True
        self.hermitian = True
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for complex dtype is not fully supported",
)
class TestMatrixRankComplexOP8(TestMatrixRankOP):
    def init_data(self):
        x_real = np.random.rand(5, 1).astype(np.float64)
        x_imag = np.random.rand(5, 1).astype(np.float64)
        self.x = x_real + 1j * x_imag
        self.tol_tensor = np.random.random([1, 4]).astype(np.float32)
        self.tol = None
        self.use_default_tol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )


class TestMatrixRankAPI(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()

        x_np = np.eye(10, dtype=np.float32)
        x_pd = paddle.to_tensor(x_np)
        rank_np = np.linalg.matrix_rank(x_np, hermitian=True)
        rank_pd = paddle.linalg.matrix_rank(x_pd, hermitian=True)
        np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

        x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
        tol_np = np.random.random([3, 4]).astype(np.float32)
        x_pd = paddle.to_tensor(x_np)
        tol_pd = paddle.to_tensor(tol_np)
        rank_np = np.linalg.matrix_rank(x_np, tol_np, hermitian=False)
        rank_pd = paddle.linalg.matrix_rank(x_pd, tol_pd, hermitian=False)
        np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

        x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
        x_pd = paddle.to_tensor(x_np)
        tol = 0.1
        rank_np = np.linalg.matrix_rank(x_np, tol, hermitian=False)
        rank_pd = paddle.linalg.matrix_rank(x_pd, tol, hermitian=False)
        np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

    def test_static(self):
        paddle.enable_static()
        places = get_places()

        for place in places:
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.random.rand(3, 4, 7, 7).astype(np.float64)
                tol_np = np.random.random([3, 4]).astype(np.float32)
                x_pd = paddle.static.data(
                    name="X", shape=[3, 4, 7, 7], dtype='float64'
                )
                tol_pd = paddle.static.data(
                    name="TolTensor", shape=[3, 4], dtype='float32'
                )
                rank_np = np.linalg.matrix_rank(x_np, tol_np, hermitian=False)
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, tol_pd, hermitian=False
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np, "TolTensor": tol_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.random.rand(3, 4, 7, 7).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=[3, 4, 7, 7], dtype='float64'
                )
                rank_np = np.linalg.matrix_rank(x_np, hermitian=True)
                rank_pd = paddle.linalg.matrix_rank(x_pd, hermitian=True)
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.random.rand(3, 4, 7, 7).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=[3, 4, 7, 7], dtype='float64'
                )
                rank_np = np.linalg.matrix_rank(x_np, 0.1, hermitian=False)
                rank_pd = paddle.linalg.matrix_rank(x_pd, 0.1, hermitian=False)
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)


class TestMatrixRankZeroSizeTensor(unittest.TestCase):

    def _get_places(self):
        return get_places()

    def _test_matrix_rank_static(self, place):
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x_valid = paddle.static.data(
                name='x_valid', shape=[2, 0, 6, 0], dtype='float32'
            )

            y_valid = paddle.linalg.matrix_rank(x_valid)

            exe = paddle.static.Executor(place)
            res_valid = exe.run(
                feed={'x_valid': np.zeros((2, 0, 6, 0), dtype='float32')},
                fetch_list=[y_valid],
            )
            self.assertEqual(res_valid[0].shape, tuple(x_valid.shape[:-2]))

    def _test_matrix_rank_dynamic_cpu(self):
        with dygraph_guard():
            paddle.set_device("cpu")
            x_valid = paddle.full((2, 0, 6, 6), 1.0, dtype='float32')
            x_valid1 = paddle.full((0, 0), 1.0, dtype='float32')
            x_valid2 = paddle.full((2, 3, 8, 0), 1.0, dtype='float32')

            y_valid = paddle.linalg.matrix_rank(x_valid)
            y_valid1 = paddle.linalg.matrix_rank(x_valid1)
            y_valid2 = paddle.linalg.matrix_rank(x_valid2)

            self.assertEqual(y_valid.shape, x_valid.shape[:-2])
            self.assertEqual(y_valid1.shape, x_valid1.shape[:-2])
            self.assertEqual(y_valid2.shape, x_valid2.shape[:-2])
            y_valid2_real = paddle.to_tensor(
                np.zeros(x_valid2.shape[:-2]).astype(np.int64)
            )
            np.testing.assert_allclose(y_valid2, y_valid2_real, rtol=1e-05)

    def _test_matrix_rank_dynamic_gpu(self):
        with dygraph_guard():
            x_valid = paddle.full((2, 0, 6, 6), 1.0, dtype='float32')
            x_valid1 = paddle.full((0, 0), 1.0, dtype='float32')
            x_valid2 = paddle.full((2, 3, 0, 7), 1.0, dtype='float32')

            y_valid = paddle.linalg.matrix_rank(x_valid)
            y_valid1 = paddle.linalg.matrix_rank(x_valid1)
            y_valid2 = paddle.linalg.matrix_rank(x_valid2)

            self.assertEqual(y_valid.shape, x_valid.shape[:-2])
            self.assertEqual(y_valid1.shape, x_valid1.shape[:-2])
            self.assertEqual(y_valid2.shape, x_valid2.shape[:-2])
            y_valid2_real = paddle.to_tensor(
                np.zeros(x_valid2.shape[:-2]).astype(np.int64)
            )
            np.testing.assert_allclose(y_valid2, y_valid2_real, rtol=1e-05)

            x_valid3 = paddle.full((2, 0, 7, 7), 1.0, dtype='float64')
            tol = paddle.full((2, 0), 1.0, dtype='float32')
            y_valid3 = paddle.linalg.matrix_rank(x_valid3, tol)
            self.assertEqual(y_valid3.shape, x_valid3.shape[:-2])

            x_valid4 = paddle.full((2, 0, 7, 7), 1.0, dtype='float64')
            atol = paddle.full((2, 0), 1.0, dtype='float32')
            y_valid4 = paddle.linalg.matrix_rank(x_valid4, atol=atol)
            self.assertEqual(y_valid4.shape, x_valid4.shape[:-2])

            x_valid5 = paddle.full((0, 7), 1.0, dtype='float64')
            tol = paddle.full((1), 1.0, dtype='float32')
            y_valid5 = paddle.linalg.matrix_rank(x_valid5, tol)
            self.assertEqual(y_valid5.shape, y_valid5.shape[:-2])

    def test_matrix_rank_tensor(self):
        for place in self._get_places():
            self._test_matrix_rank_static(place)
        self._test_matrix_rank_dynamic_cpu()
        self._test_matrix_rank_dynamic_gpu()


if __name__ == '__main__':
    unittest.main()
