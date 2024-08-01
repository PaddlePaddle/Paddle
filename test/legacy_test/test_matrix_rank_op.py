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

import os
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle import _C_ops, base, static
from paddle.base import core
from paddle.base.data_feeder import check_type, check_variable_and_dtype
from paddle.common_ops_import import Variable
from paddle.framework import LayerHelper, in_dynamic_or_pir_mode
from paddle.pir_utils import test_with_pir_api
from paddle.tensor.manipulation import cast

paddle.enable_static()
SEED = 2049
np.random.seed(SEED)


def matrix_rank_wraper(x, tol=None, use_default_tol=True, hermitian=False):
    if in_dynamic_or_pir_mode():
        if isinstance(tol, (Variable, paddle.pir.Value)):
            if tol.dtype != x.dtype:
                tol_tensor = cast(tol, x.dtype)
            else:
                tol_tensor = tol
            use_default_tol = False
            return _C_ops.matrix_rank_tol(
                x, tol_tensor, use_default_tol, hermitian
            )

        if tol is None:
            tol_attr = 0.0
            use_default_tol = True
        else:
            tol_attr = float(tol)
            use_default_tol = False
        return _C_ops.matrix_rank(x, tol_attr, use_default_tol, hermitian)
    else:
        inputs = {}
        attrs = {}
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'matrix_rank')
        inputs['X'] = x
        if tol is None:
            attrs['use_default_tol'] = True
        elif isinstance(tol, Variable):
            attrs['use_default_tol'] = False
            if tol.dtype != x.dtype:
                inputs['TolTensor'] = cast(tol, x.dtype)
            else:
                inputs['TolTensor'] = tol
        else:
            check_type(tol, 'tol', float, 'matrix_rank')
            attrs['use_default_tol'] = False
            attrs['tol'] = tol
        check_type(hermitian, 'hermitian', bool, 'matrix_rank')
        attrs['hermitian'] = hermitian

        helper = LayerHelper('matrix_rank', **locals())
        out = helper.create_variable_for_type_inference(dtype='int32')
        helper.append_op(
            type='matrix_rank', inputs=inputs, outputs={'Out': out}, attrs=attrs
        )
        return out


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


def matrix_rank_atol_rtol_wraper(
    x, tol=None, atol=None, rtol=None, use_default_tol=True, hermitian=False
):
    return paddle.linalg.matrix_rank(x, tol, hermitian, atol, rtol)


class TestMatrixRankAtolRtolOP(OpTest):
    def setUp(self):
        self.python_api = matrix_rank_atol_rtol_wraper
        self.op_type = "matrix_rank_atol_rtol"
        self.use_default_tol = False
        self.init_data()
        self.process_data()
        self.inputs = {'x': self.x}
        if self.use_atol_rtol:
            self.inputs['atol'] = self.atol
            self.inputs['rtol'] = self.rtol
        else:
            self.inputs['tol'] = self.tol
        self.attrs = {'hermitian': self.hermitian}
        self.attrs["use_default_tol"] = self.use_default_tol
        self.outputs = {'out': self.out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def init_data(self):
        self.x = np.eye(3, dtype=np.float32)
        self.tol = 0.1
        self.atol = None
        self.rtol = None
        self.use_default_tol = False
        self.hermitian = True
        self.use_atol_rtol = False

    def process_data(self):
        if self.use_atol_rtol:
            self.out = np_matrix_rank_atol_rtol(
                self.x, self.atol, self.rtol, self.hermitian
            )

            if self.rtol is None:
                self.rtol = np.full([], 0.0, self.x.dtype)
                if (self.atol is None) or (
                    isinstance(self.atol, (float, int)) and self.atol == 0
                ):
                    self.use_default_tol = True
            if self.atol is None:
                self.atol = np.full([], 0.0, self.x.dtype)

            if isinstance(self.atol, (float, int)):
                self.atol = np.full([], self.atol, self.x.dtype)
            if isinstance(self.rtol, (float, int)):
                self.rtol = np.full([], self.rtol, self.x.dtype)
            self.atol, self.rtol = np.broadcast_arrays(self.atol, self.rtol)
        else:
            self.out = np.linalg.matrix_rank(self.x, self.tol, self.hermitian)

            if self.tol is None:
                self.tol = np.full([], 0.0, self.x.dtype)
                self.use_default_tol = True
            if isinstance(self.tol, (float, int)):
                self.tol = np.full([], self.tol, self.x.dtype)


class TestMatrixRankAtolRtolOP1(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.eye(3, dtype=np.float32)
        self.tol = None
        self.atol = None
        self.rtol = None
        self.use_default_tol = False
        self.hermitian = True
        self.use_atol_rtol = False


class TestMatrixRankAtolRtolOP2(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.tol = np.random.random([3, 4]).astype(self.x.dtype)
        self.atol = None
        self.rtol = None
        self.use_default_tol = False
        self.hermitian = False
        self.use_atol_rtol = False


class TestMatrixRankAtolRtolOP3(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(1, 10).astype(np.float32)
        self.tol = None
        self.atol = None
        self.rtol = None
        self.use_default_tol = False
        self.hermitian = False
        self.use_atol_rtol = False


class TestMatrixRankAtolRtolOP4(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(5, 1).astype(np.float64)
        self.tol = np.random.random([1, 4]).astype(self.x.dtype)
        self.atol = None
        self.rtol = None
        self.use_default_tol = False
        self.hermitian = False
        self.use_atol_rtol = False


class TestMatrixRankAtolRtolOP5(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.tol = np.random.random([200, 200]).astype(self.x.dtype)
        self.atol = None
        self.rtol = None
        self.use_default_tol = False
        self.hermitian = True
        self.use_atol_rtol = False


class TestMatrixRankAtolRtolOP6(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.eye(3, dtype=np.float32)
        self.tol = None
        self.atol = 0.05
        self.rtol = None
        self.use_default_tol = False
        self.hermitian = True
        self.use_atol_rtol = True


class TestMatrixRankAtolRtolOP7(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.tol = None
        self.atol = np.random.random([3, 4]).astype(self.x.dtype)
        self.rtol = None
        self.use_default_tol = False
        self.hermitian = False
        self.use_atol_rtol = True


class TestMatrixRankAtolRtolOP8(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.tol = None
        self.atol = None
        self.rtol = np.random.random([3, 4]).astype(self.x.dtype)
        self.use_default_tol = False
        self.hermitian = False
        self.use_atol_rtol = True


class TestMatrixRankAtolRtolOP9(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(1, 10).astype(np.float32)
        self.tol = None
        self.atol = 0.2
        self.rtol = 1.1
        self.use_default_tol = False
        self.hermitian = False
        self.use_atol_rtol = True


class TestMatrixRankAtolRtolOP10(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(5, 1).astype(np.float64)
        self.tol = None
        self.atol = np.random.random([1, 4]).astype(self.x.dtype)
        self.rtol = np.random.random([1, 4]).astype(self.x.dtype)
        self.use_default_tol = False
        self.hermitian = False
        self.use_atol_rtol = True


class TestMatrixRankAtolRtolOP11(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.tol = None
        self.atol = np.random.random([200, 200]).astype(self.x.dtype)
        self.rtol = 0.8
        self.use_default_tol = False
        self.hermitian = True
        self.use_atol_rtol = True


class TestMatrixRankAtolRtolOP12(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.tol = None
        self.atol = np.random.random([200, 1]).astype(self.x.dtype)
        self.rtol = np.random.random([200, 200]).astype(self.x.dtype)
        self.use_default_tol = False
        self.hermitian = True
        self.use_atol_rtol = True


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
            atol_np = np.random.random([3, 1]).astype(np.float32)
            rtol_np = np.random.random([3, 1]).astype(np.float32)
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
                x_np = np.eye(10).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=x_np.shape, dtype='float64'
                )
                atol_np = np.array([0.02]).astype(np.float32)
                atol_pd = paddle.static.data(
                    name="Atol", shape=atol_np.shape, dtype='float32'
                )
                rtol_np = np.array([0.02]).astype(np.float32)
                rtol_pd = paddle.static.data(
                    name="Rtol", shape=rtol_np.shape, dtype='float32'
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
