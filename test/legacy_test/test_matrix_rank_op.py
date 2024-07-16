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
from op_test import OpTest

import paddle
from paddle import base, static
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

paddle.enable_static()
SEED = 2049
np.random.seed(SEED)


def matrix_rank_wraper(
    x,
    tol=None,
    use_default_atol=True,
    use_default_rtol=True,
    hermitian=False,
    atol=None,
    rtol=None,
):
    return paddle.linalg.matrix_rank(x, tol, hermitian)


class TestMatrixRankOP(OpTest):
    def setUp(self):
        self.python_api = matrix_rank_wraper
        self.op_type = "matrix_rank"
        self.init_data()
        self.inputs = {'X': self.x}
        self.attrs = {'hermitian': self.hermitian}
        if self.tol_tensor is not None:
            self.inputs["TolTensor"] = self.tol_tensor
        if self.tol is not None:
            self.attrs["tol"] = self.tol
        self.attrs["atol"] = self.atol
        self.attrs["rtol"] = self.rtol
        self.attrs["use_default_atol"] = self.use_default_atol
        self.attrs["use_default_rtol"] = self.use_default_rtol
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def init_data(self):
        self.x = np.eye(3, dtype=np.float32)
        self.tol_tensor = None
        self.tol = 0.1
        self.atol = 0.0
        self.rtol = 0.0
        self.use_default_atol = False
        self.use_default_rtol = True
        self.hermitian = True
        self.out = np.linalg.matrix_rank(self.x, self.tol, self.hermitian)


class TestMatrixRankOP1(TestMatrixRankOP):
    def init_data(self):
        self.x = np.eye(3, k=1, dtype=np.float64)
        self.tol_tensor = None
        self.tol = None
        self.use_default_atol = True
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )
        self.atol = 0.0
        self.rtol = 0.0
        self.use_default_rtol = True


class TestMatrixRankOP2(TestMatrixRankOP):
    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.tol_tensor = np.random.random([3, 4]).astype(self.x.dtype)
        self.tol = None
        self.use_default_atol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )
        self.atol = 0.0
        self.rtol = 0.0
        self.use_default_rtol = True


class TestMatrixRankOP3(TestMatrixRankOP):
    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.tol_tensor = None
        self.tol = None
        self.use_default_atol = True
        self.hermitian = True
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )
        self.atol = 0.0
        self.rtol = 0.0
        self.use_default_rtol = True


class TestMatrixRankOP4(TestMatrixRankOP):
    def init_data(self):
        self.x = np.random.rand(1, 10).astype(np.float32)
        self.tol_tensor = None
        self.tol = None
        self.use_default_atol = True
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )
        self.atol = 0.0
        self.rtol = 0.0
        self.use_default_rtol = True


class TestMatrixRankOP5(TestMatrixRankOP):
    def init_data(self):
        self.x = np.random.rand(5, 1).astype(np.float64)
        self.tol_tensor = np.random.random([1, 4]).astype(self.x.dtype)
        self.tol = None
        self.use_default_atol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )
        self.atol = 0.0
        self.rtol = 0.0
        self.use_default_rtol = True


class TestMatrixRankOP6(TestMatrixRankOP):
    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.tol_tensor = None
        self.tol = None
        self.use_default_atol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )
        self.atol = 0.0
        self.rtol = 0.0
        self.use_default_rtol = True


class TestMatrixRankOP7(TestMatrixRankOP):
    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.tol_tensor = np.random.random([200, 200]).astype(self.x.dtype)
        self.tol = None
        self.use_default_atol = True
        self.hermitian = True
        self.out = np.linalg.matrix_rank(
            self.x, self.tol_tensor, self.hermitian
        )
        self.atol = 0.0
        self.rtol = 0.0
        self.use_default_rtol = True


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

    @test_with_pir_api
    def test_static(self):
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

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


def np_matrix_rank_atol_rtol(x, atol=None, rtol=None, hermitian=False):
    if (atol is None) and (rtol is None):
        return np.linalg.matrix_rank(x, hermitian=hermitian)
    if atol is None:
        atol = 0.0
    if rtol is None:
        rtol = 0.0
    _, sv, _ = np.linalg.svd(x)
    sv_max = sv.max(axis=-1)
    tol = np.maximum(atol, rtol * sv_max)
    return np.linalg.matrix_rank(x, tol, hermitian=hermitian)


class TestMatrixRankAtolRtolAPI(unittest.TestCase):
    def test_dygraph(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            paddle.disable_static(place)

            # atol: float, rtol: None
            x_np = np.eye(10, dtype=np.float32)
            x_pd = paddle.to_tensor(x_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=0.015, rtol=None, hermitian=True
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=True, atol=0.015, rtol=None
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: None, rtol: float
            x_np = np.eye(10, dtype=np.float32)
            x_pd = paddle.to_tensor(x_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=None, rtol=1.1, hermitian=True
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=True, atol=None, rtol=1.1
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: float, rtol: float
            x_np = np.eye(10, dtype=np.float32)
            x_pd = paddle.to_tensor(x_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=0.2, rtol=0.05, hermitian=True
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=True, atol=0.2, rtol=0.05
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: tensor, rtol: float
            x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
            atol_np = np.random.random([3, 4]).astype(np.float32)
            x_pd = paddle.to_tensor(x_np)
            atol_pd = paddle.to_tensor(atol_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=atol_np, rtol=0.01, hermitian=False
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=False, atol=atol_pd, rtol=0.01
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: float, rtol: tensor
            x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
            rtol_np = np.random.random([3, 4]).astype(np.float32)
            x_pd = paddle.to_tensor(x_np)
            rtol_pd = paddle.to_tensor(rtol_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=0.01, rtol=rtol_np, hermitian=False
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=False, atol=0.01, rtol=rtol_pd
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: tensor, rtol: tensor
            x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
            atol_np = np.random.random([3, 4]).astype(np.float32)
            rtol_np = np.random.random([3, 4]).astype(np.float32)
            x_pd = paddle.to_tensor(x_np)
            atol_pd = paddle.to_tensor(atol_np)
            rtol_pd = paddle.to_tensor(rtol_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=atol_np, rtol=rtol_np, hermitian=False
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=False, atol=atol_pd, rtol=rtol_pd
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: tensor, rtol: tensor; broadcast_shape
            x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
            atol_np = np.random.random([3, 4]).astype(np.float32)
            rtol_np = np.random.random([3, 4]).astype(np.float32)
            x_pd = paddle.to_tensor(x_np)
            atol_pd = paddle.to_tensor(atol_np)
            rtol_pd = paddle.to_tensor(rtol_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=atol_np, rtol=rtol_np, hermitian=False
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=False, atol=atol_pd, rtol=rtol_pd
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

    @test_with_pir_api
    def test_static(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            # atol: float, rtol: None
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.eye(10).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=x_np.shape, dtype='float64'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=0.015, rtol=None, hermitian=True
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=True, atol=0.015, rtol=None
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: None, rtol: float
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.eye(10).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=x_np.shape, dtype='float64'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=None, rtol=1.1, hermitian=True
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=True, atol=None, rtol=1.1
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: float, rtol: float
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.eye(10).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=x_np.shape, dtype='float64'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=0.05, rtol=0.2, hermitian=True
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=True, atol=0.05, rtol=0.2
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: tensor, rtol: float
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=[3, 4, 7, 8], dtype='float64'
                )
                atol_np = np.random.random([3, 4]).astype(np.float32)
                atol_pd = paddle.static.data(
                    name="Atol", shape=[3, 4], dtype='float32'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=atol_np, rtol=0.02, hermitian=False
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=False, atol=atol_pd, rtol=0.02
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np, "Atol": atol_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: float, rtol: tensor
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=[3, 4, 7, 8], dtype='float64'
                )
                rtol_np = np.random.random([3, 4]).astype(np.float32)
                rtol_pd = paddle.static.data(
                    name="Rtol", shape=[3, 4], dtype='float32'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=0.02, rtol=rtol_np, hermitian=False
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=False, atol=0.02, rtol=rtol_pd
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np, "Rtol": rtol_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: tensor, rtol: tensor
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.random.rand(3, 4, 7, 7).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=[3, 4, 7, 7], dtype='float64'
                )
                atol_np = np.random.random([3, 4]).astype(np.float32)
                atol_pd = paddle.static.data(
                    name="Atol", shape=[3, 4], dtype='float32'
                )
                rtol_np = np.random.random([3, 4]).astype(np.float32)
                rtol_pd = paddle.static.data(
                    name="Rtol", shape=[3, 4], dtype='float32'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=atol_np, rtol=rtol_np, hermitian=True
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=True, atol=atol_pd, rtol=rtol_pd
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np, "Atol": atol_np, "Rtol": rtol_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)


class TestMatrixRankError(unittest.TestCase):
    def test_errors(self):
        x = paddle.eye(10)
        with self.assertRaises(ValueError):
            paddle.linalg.matrix_rank(x, tol=0.2, hermitian=True, atol=0.2)

        with self.assertRaises(ValueError):
            paddle.linalg.matrix_rank(x, tol=0.2, hermitian=True, rtol=0.2)


if __name__ == '__main__':
    unittest.main()
