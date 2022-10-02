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

from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()
SEED = 2049
np.random.seed(SEED)


def matrix_rank_wraper(x, tol=None, use_default_tol=True, hermitian=False):
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
        self.attrs["use_default_tol"] = self.use_default_tol
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_eager=True)

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
        self.out = np.linalg.matrix_rank(self.x, self.tol_tensor,
                                         self.hermitian)


class TestMatrixRankOP2(TestMatrixRankOP):

    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.tol_tensor = np.random.random([3, 4]).astype(self.x.dtype)
        self.tol = None
        self.use_default_tol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(self.x, self.tol_tensor,
                                         self.hermitian)


class TestMatrixRankOP3(TestMatrixRankOP):

    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = True
        self.hermitian = True
        self.out = np.linalg.matrix_rank(self.x, self.tol_tensor,
                                         self.hermitian)


class TestMatrixRankOP4(TestMatrixRankOP):

    def init_data(self):
        self.x = np.random.rand(1, 10).astype(np.float32)
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = True
        self.hermitian = False
        self.out = np.linalg.matrix_rank(self.x, self.tol_tensor,
                                         self.hermitian)


class TestMatrixRankOP5(TestMatrixRankOP):

    def init_data(self):
        self.x = np.random.rand(5, 1).astype(np.float64)
        self.tol_tensor = np.random.random([1, 4]).astype(self.x.dtype)
        self.tol = None
        self.use_default_tol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(self.x, self.tol_tensor,
                                         self.hermitian)


class TestMatrixRankOP6(TestMatrixRankOP):

    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.tol_tensor = None
        self.tol = None
        self.use_default_tol = False
        self.hermitian = False
        self.out = np.linalg.matrix_rank(self.x, self.tol_tensor,
                                         self.hermitian)


class TestMatrixRankOP7(TestMatrixRankOP):

    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.tol_tensor = np.random.random([200, 200]).astype(self.x.dtype)
        self.tol = None
        self.use_default_tol = True
        self.hermitian = True
        self.out = np.linalg.matrix_rank(self.x, self.tol_tensor,
                                         self.hermitian)


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
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                x_np = np.random.rand(3, 4, 7, 7).astype(np.float64)
                tol_np = np.random.random([3, 4]).astype(np.float32)
                x_pd = paddle.fluid.data(name="X",
                                         shape=[3, 4, 7, 7],
                                         dtype='float64')
                tol_pd = paddle.fluid.data(name="TolTensor",
                                           shape=[3, 4],
                                           dtype='float32')
                rank_np = np.linalg.matrix_rank(x_np, tol_np, hermitian=False)
                rank_pd = paddle.linalg.matrix_rank(x_pd,
                                                    tol_pd,
                                                    hermitian=False)
                exe = fluid.Executor(place)
                fetches = exe.run(fluid.default_main_program(),
                                  feed={
                                      "X": x_np,
                                      "TolTensor": tol_np
                                  },
                                  fetch_list=[rank_pd])
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                x_np = np.random.rand(3, 4, 7, 7).astype(np.float64)
                x_pd = paddle.fluid.data(name="X",
                                         shape=[3, 4, 7, 7],
                                         dtype='float64')
                rank_np = np.linalg.matrix_rank(x_np, hermitian=True)
                rank_pd = paddle.linalg.matrix_rank(x_pd, hermitian=True)
                exe = fluid.Executor(place)
                fetches = exe.run(fluid.default_main_program(),
                                  feed={"X": x_np},
                                  fetch_list=[rank_pd])
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                x_np = np.random.rand(3, 4, 7, 7).astype(np.float64)
                x_pd = paddle.fluid.data(name="X",
                                         shape=[3, 4, 7, 7],
                                         dtype='float64')
                rank_np = np.linalg.matrix_rank(x_np, 0.1, hermitian=False)
                rank_pd = paddle.linalg.matrix_rank(x_pd, 0.1, hermitian=False)
                exe = fluid.Executor(place)
                fetches = exe.run(fluid.default_main_program(),
                                  feed={"X": x_np},
                                  fetch_list=[rank_pd])
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
