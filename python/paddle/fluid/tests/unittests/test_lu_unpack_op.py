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

from __future__ import print_function
from op_test import OpTest
import unittest
import itertools
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import scipy
import scipy.linalg
import copy


def scipy_lu_unpack(A):
    shape = A.shape
    if len(shape) == 2:
        return scipy.linalg.lu(A)
    else:
        preshape = shape[:-2]
        batchsize = np.product(shape) // (shape[-2] * shape[-1])
        Plst = []
        Llst = []
        Ulst = []

        NA = A.reshape((-1, shape[-2], shape[-1]))
        for b in range(batchsize):
            As = NA[b]
            P, L, U = scipy.linalg.lu(As)

            pshape = P.shape
            lshape = L.shape
            ushape = U.shape

            Plst.append(P)
            Llst.append(L)
            Ulst.append(U)

        return np.array(Plst).reshape(preshape + pshape), np.array(
            Llst).reshape(preshape + lshape), np.array(Ulst).reshape(preshape +
                                                                     ushape)


def Pmat_to_perm(Pmat_org, cut):
    Pmat = copy.deepcopy(Pmat_org)
    shape = Pmat.shape
    rows = shape[-2]
    cols = shape[-1]
    batchsize = max(1, np.product(shape[:-2]))
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
    Pivot = np.array(permmat).reshape(list(shape[:-2]) + [
        rows,
    ]) + 1

    return Pivot[..., :cut]


def perm_to_Pmat(perm, dim):
    pshape = perm.shape
    bs = int(np.product(perm.shape[:-1]).item())
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
    return np.array(oneslst).reshape(list(pshape[:-1]) + [dim, dim])


# m > n
class TestLU_UnpackOp(OpTest):
    """
    case 1
    """

    def config(self):
        self.x_shape = [2, 12, 10]
        self.unpack_ludata = True
        self.unpack_pivots = True
        self.dtype = "float64"

    def set_output(self, A):
        sP, sL, sU = scipy_lu_unpack(A)
        self.L = sL
        self.U = sU
        self.P = sP

    def setUp(self):
        self.op_type = "lu_unpack"
        self.python_api = paddle.tensor.linalg.lu_unpack
        self.python_out_sig = ["Pmat", "L", "U"]
        self.config()
        x = np.random.random(self.x_shape).astype(self.dtype)
        if paddle.in_dynamic_mode():
            xt = paddle.to_tensor(x)
            lu, pivots = paddle.linalg.lu(xt)
            lu = lu.numpy()
            pivots = pivots.numpy()
        else:
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                place = fluid.CPUPlace()
                if core.is_compiled_with_cuda():
                    place = fluid.CUDAPlace(0)
                xv = paddle.fluid.data(name="input",
                                       shape=self.x_shape,
                                       dtype=self.dtype)
                lu, p = paddle.linalg.lu(xv)
                exe = fluid.Executor(place)
                fetches = exe.run(fluid.default_main_program(),
                                  feed={"input": x},
                                  fetch_list=[lu, p])
                lu, pivots = fetches[0], fetches[1]

        self.inputs = {'X': lu, 'Pivots': pivots}

        self.attrs = {
            'unpack_ludata': self.unpack_ludata,
            'unpack_pivots': self.unpack_pivots
        }
        self.set_output(x)
        self.outputs = {
            'Pmat': self.P,
            'L': self.L,
            'U': self.U,
        }

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], ['L', 'U'], check_eager=True)


# m = n
class TestLU_UnpackOp2(TestLU_UnpackOp):
    """
    case 2
    """

    def config(self):
        self.x_shape = [2, 10, 10]
        self.unpack_ludata = True
        self.unpack_pivots = True
        self.dtype = "float64"


# m < n
class TestLU_UnpackOp3(TestLU_UnpackOp):
    """
    case 3
    """

    def config(self):
        self.x_shape = [2, 10, 12]
        self.unpack_ludata = True
        self.unpack_pivots = True
        self.dtype = "float64"


class TestLU_UnpackAPI(unittest.TestCase):

    def setUp(self):
        np.random.seed(2022)

    def test_dygraph(self):

        def run_lu_unpack_dygraph(shape, dtype):
            if dtype == "float32":
                np_dtype = np.float32
            elif dtype == "float64":
                np_dtype = np.float64
            a = np.random.rand(*shape).astype(np_dtype)
            m = a.shape[-2]
            n = a.shape[-1]
            min_mn = min(m, n)

            places = [fluid.CPUPlace()]
            if core.is_compiled_with_cuda():
                places.append(fluid.CUDAPlace(0))
            for place in places:
                paddle.disable_static(place)

                x = paddle.to_tensor(a, dtype=dtype)
                sP, sL, sU = scipy_lu_unpack(a)
                LU, P = paddle.linalg.lu(x)
                pP, pL, pU = paddle.linalg.lu_unpack(LU, P)

                np.testing.assert_allclose(sU, pU, rtol=1e-05, atol=1e-05)
                np.testing.assert_allclose(sL, pL, rtol=1e-05, atol=1e-05)
                np.testing.assert_allclose(sP, pP, rtol=1e-05, atol=1e-05)

        tensor_shapes = [
            (3, 5),
            (5, 5),
            (5, 3),  # 2-dim Tensors
            (2, 3, 5),
            (3, 5, 5),
            (4, 5, 3),  # 3-dim Tensors
            (2, 5, 3, 5),
            (3, 5, 5, 5),
            (4, 5, 5, 3)  # 4-dim Tensors
        ]
        dtypes = ["float32", "float64"]
        for tensor_shape, dtype in itertools.product(tensor_shapes, dtypes):
            run_lu_unpack_dygraph(tensor_shape, dtype)

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

            places = [fluid.CPUPlace()]
            if core.is_compiled_with_cuda():
                places.append(fluid.CUDAPlace(0))
            for place in places:
                with fluid.program_guard(fluid.Program(), fluid.Program()):
                    sP, sL, sU = scipy_lu_unpack(a)

                    x = paddle.fluid.data(name="input",
                                          shape=shape,
                                          dtype=dtype)
                    lu, p = paddle.linalg.lu(x)
                    pP, pL, pU = paddle.linalg.lu_unpack(lu, p)
                    exe = fluid.Executor(place)
                    fetches = exe.run(fluid.default_main_program(),
                                      feed={"input": a},
                                      fetch_list=[pP, pL, pU])
                    np.testing.assert_allclose(fetches[0],
                                               sP,
                                               rtol=1e-05,
                                               atol=1e-05)
                    np.testing.assert_allclose(fetches[1],
                                               sL,
                                               rtol=1e-05,
                                               atol=1e-05)
                    np.testing.assert_allclose(fetches[2],
                                               sU,
                                               rtol=1e-05,
                                               atol=1e-05)

        tensor_shapes = [
            (3, 5),
            (5, 5),
            (5, 3),  # 2-dim Tensors
            (2, 3, 5),
            (3, 5, 5),
            (4, 5, 3),  # 3-dim Tensors
            (2, 5, 3, 5),
            (3, 5, 5, 5),
            (4, 5, 5, 3)  # 4-dim Tensors
        ]
        dtypes = ["float32", "float64"]
        for tensor_shape, dtype in itertools.product(tensor_shapes, dtypes):
            run_lu_static(tensor_shape, dtype)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
