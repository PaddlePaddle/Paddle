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

import copy
import itertools
import os
import unittest

import numpy as np
import scipy
import scipy.linalg
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import core


def scipy_lu(A, pivot):
    shape = A.shape
    if len(shape) == 2:
        return scipy.linalg.lu(A, permute_l=not pivot)
    else:
        preshape = shape[:-2]
        batchsize = np.prod(shape) // (shape[-2] * shape[-1])
        PP = []
        PL = []
        PU = []
        NA = A.reshape((-1, shape[-2], shape[-1]))
        for b in range(batchsize):
            P, L, U = scipy.linalg.lu(NA[b], permute_l=not pivot)
            pshape = P.shape
            lshape = L.shape
            ushape = U.shape
            PP.append(P)
            PL.append(L)
            PU.append(U)
        return (
            np.array(PP).reshape(preshape + pshape),
            np.array(PL).reshape(preshape + lshape),
            np.array(PU).reshape(preshape + ushape),
        )


def Pmat_to_perm(Pmat_org, cut):
    Pmat = copy.deepcopy(Pmat_org)
    shape = Pmat.shape
    rows = shape[-2]
    cols = shape[-1]
    batchsize = max(1, np.prod(shape[:-2]))
    P = Pmat.reshape(batchsize, rows, cols)
    permmat = []
    for b in range(batchsize):
        permlst = []
        sP = P[b]
        for c in range(min(rows, cols)):
            idx = np.argmax(sP[:, c])
            permlst.append(idx)
            tmp = copy.deepcopy(sP[c, :])
            sP[c, :] = sP[idx, :]
            sP[idx, :] = tmp

        permmat.append(permlst)
    Pivot = (
        np.array(permmat).reshape(
            [
                *shape[:-2],
                rows,
            ]
        )
        + 1
    )
    return Pivot[..., :cut]


def perm_to_Pmat(perm, dim):
    pshape = perm.shape
    bs = int(np.prod(perm.shape[:-1]).item())
    perm = perm.reshape((bs, pshape[-1]))
    oneslst = []
    for i in range(bs):
        idlst = np.arange(dim)
        perm_item = perm[i, :]
        for idx, p in enumerate(perm_item - 1):
            temp = idlst[idx]
            idlst[idx] = idlst[p]
            idlst[p] = temp

        ones = paddle.eye(dim)
        nmat = paddle.scatter(ones, paddle.to_tensor(idlst), ones)
        oneslst.append(nmat)
    return np.array(oneslst).reshape([*pshape[:-1], dim, dim])


# m < n
class TestLUOp(OpTest):
    """
    case 1
    """

    def config(self):
        self.x_shape = [3, 10, 12]
        self.pivot = True
        self.get_infos = True
        self.dtype = "float64"

    def set_output(self):
        X = self.inputs['X']
        sP, sl, sU = scipy_lu(X, self.pivot)
        sL = np.tril(sl, -1)
        ashape = np.array(X.shape)
        lshape = np.array(sL.shape)
        ushape = np.array(sU.shape)

        lpad = (len(sL.shape) - 2) * [(0, 0)] + [
            (0, (ashape - lshape)[-2]),
            (0, (ashape - lshape)[-1]),
        ]
        upad = (len(sU.shape) - 2) * [(0, 0)] + [
            (0, (ashape - ushape)[-2]),
            (0, (ashape - ushape)[-1]),
        ]

        NsL = np.pad(sL, lpad)
        NsU = np.pad(sU, upad)
        NLU = NsL + NsU
        self.output = NLU
        self.Pivots = Pmat_to_perm(sP, min(ashape[-2], ashape[-1]))
        self.Infos = (
            np.zeros(self.x_shape[:-2]) if len(X.shape) > 2 else np.array(0)
        )

    def setUp(self):
        self.op_type = "lu"
        self.python_api = paddle.tensor.linalg.lu
        self.python_out_sig = ["Out", "Pivots"]
        self.config()

        self.inputs = {'X': np.random.random(self.x_shape).astype(self.dtype)}
        self.attrs = {'pivots': self.pivot}
        self.set_output()
        self.outputs = {
            'Out': self.output,
            'Pivots': self.Pivots,
            'Infos': self.Infos,
        }

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], ['Out'], check_pir=True)


# m = n 2D
class TestLUOp2(TestLUOp):
    """
    case 2
    """

    def config(self):
        self.x_shape = [10, 10]
        self.pivot = True
        self.get_infos = True
        self.dtype = "float64"


# m > n
class TestLUOp3(TestLUOp):
    """
    case 3
    """

    def config(self):
        self.x_shape = [2, 12, 10]
        self.pivot = True
        self.get_infos = True
        self.dtype = "float64"


class TestLUAPI(unittest.TestCase):
    def test_dygraph(self):
        def run_lu_dygraph(shape, dtype):
            if dtype == "float32":
                np_dtype = np.float32
            elif dtype == "float64":
                np_dtype = np.float64
            np.random.seed(1024)
            a = np.random.rand(*shape).astype(np_dtype)
            m = a.shape[-2]
            n = a.shape[-1]
            min_mn = min(m, n)
            pivot = True

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
                paddle.disable_static(place)
                batch_size = a.size // (a.shape[-1] * a.shape[-2])
                x = paddle.to_tensor(a, dtype=dtype)
                sP, sl, sU = scipy_lu(a, pivot)
                sL = np.tril(sl, -1)
                LU, P, Info = paddle.linalg.lu(x, pivot=pivot, get_infos=True)
                m, n = LU.shape[-2], LU.shape[-1]
                tril = np.tril(LU, -1)[..., :m, :m]
                triu = np.triu(LU)[..., :n, :n]
                mtp = Pmat_to_perm(sP, min(m, n))
                nP = perm_to_Pmat(P, sP.shape[-1])

                np.testing.assert_allclose(sU, triu, rtol=1e-05, atol=1e-05)
                np.testing.assert_allclose(sL, tril, rtol=1e-05, atol=1e-05)
                np.testing.assert_allclose(P, mtp, rtol=1e-05, atol=1e-05)
                np.testing.assert_allclose(nP, sP, rtol=1e-05, atol=1e-05)

        tensor_shapes = [
            (3, 5),
            (5, 5),
            (5, 3),  # 2-dim Tensors
            (2, 3, 5),
            (3, 5, 5),
            (4, 5, 3),  # 3-dim Tensors
            (2, 5, 3, 5),
            (3, 5, 5, 5),
            (4, 5, 5, 3),  # 4-dim Tensors
        ]
        dtypes = ["float32", "float64"]
        for tensor_shape, dtype in itertools.product(tensor_shapes, dtypes):
            run_lu_dygraph(tensor_shape, dtype)

    def test_static(self):
        paddle.enable_static()

        def run_lu_static(shape, dtype):
            if dtype == "float32":
                np_dtype = np.float32
            elif dtype == "float64":
                np_dtype = np.float64
            a = np.random.rand(*shape).astype(np_dtype)
            m = a.shape[-2]
            n = a.shape[-1]
            min_mn = min(m, n)
            pivot = True

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
                    batch_size = a.size // (a.shape[-1] * a.shape[-2])
                    sP, sl, sU = scipy_lu(a, pivot)
                    sL = np.tril(sl, -1)
                    ashape = np.array(a.shape)
                    lshape = np.array(sL.shape)
                    ushape = np.array(sU.shape)

                    lpad = (len(sL.shape) - 2) * [(0, 0)] + [
                        (0, (ashape - lshape)[-2]),
                        (0, (ashape - lshape)[-1]),
                    ]
                    upad = (len(sU.shape) - 2) * [(0, 0)] + [
                        (0, (ashape - ushape)[-2]),
                        (0, (ashape - ushape)[-1]),
                    ]

                    NsL = np.pad(sL, lpad)
                    NsU = np.pad(sU, upad)
                    NLU = NsL + NsU

                    x = paddle.static.data(
                        name="input", shape=shape, dtype=dtype
                    )
                    lu, p = paddle.linalg.lu(x, pivot=pivot)
                    exe = base.Executor(place)
                    fetches = exe.run(
                        feed={"input": a},
                        fetch_list=[lu, p],
                    )
                    np.testing.assert_allclose(
                        fetches[0], NLU, rtol=1e-05, atol=1e-05
                    )

        tensor_shapes = [
            (3, 5),
            (5, 5),
            (5, 3),  # 2-dim Tensors
            (2, 3, 5),
            (3, 5, 5),
            (4, 5, 3),  # 3-dim Tensors
            (2, 5, 3, 5),
            (3, 5, 5, 5),
            (4, 5, 5, 3),  # 4-dim Tensors
        ]
        dtypes = ["float32", "float64"]
        for tensor_shape, dtype in itertools.product(tensor_shapes, dtypes):
            run_lu_static(tensor_shape, dtype)


class TestLUAPIError(unittest.TestCase):
    def test_errors(self):
        with paddle.base.dygraph.guard():
            # The size of input in lu should not be 0.
            def test_0_size():
                array = np.array([], dtype=np.float32)
                x = paddle.to_tensor(
                    np.reshape(array, [0, 0, 0]), dtype='float32'
                )
                paddle.linalg.lu(x, get_infos=True)

            self.assertRaises(ValueError, test_0_size)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
