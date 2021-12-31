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


def scipy_lu(A, pivot):
    shape = A.shape
    if len(shape) == 2:
        return scipy.linalg.lu(A, permute_l=not pivot)
    else:
        preshape = shape[:-2]
        batchsize = np.product(shape) // (shape[-2] * shape[-1])
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
        return np.array(PP).reshape(preshape + pshape), np.array(PL).reshape(
            preshape + lshape), np.array(PU).reshape(preshape + ushape)


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
    Pivot = np.array(permmat).reshape(list(shape[:-2]) + [rows, ]) + 1
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

        lpad = (len(sL.shape) - 2) * [(0, 0)] + list((
            (0, (ashape - lshape)[-2]), (0, (ashape - lshape)[-1])))
        upad = (len(sU.shape) - 2) * [(0, 0)] + list((
            (0, (ashape - ushape)[-2]), (0, (ashape - ushape)[-1])))

        NsL = np.pad(sL, lpad)
        NsU = np.pad(sU, upad)
        NLU = NsL + NsU
        self.output = NLU
        self.Pivots = Pmat_to_perm(sP, min(ashape[-2], ashape[-1]))
        self.Infos = np.zeros(self.x_shape[:-2]) if len(
            X.shape) > 2 else np.array([0])

    def setUp(self):
        self.op_type = "lu"
        self.config()

        self.inputs = {'X': np.random.random(self.x_shape).astype(self.dtype)}
        self.attrs = {'pivots': self.pivot}
        self.set_output()
        self.outputs = {
            'Out': self.output,
            'Pivots': self.Pivots,
            'Infos': self.Infos
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['Out'])


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


if __name__ == "__main__":
    unittest.main()
