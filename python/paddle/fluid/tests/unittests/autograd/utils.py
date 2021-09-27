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

import numpy as np
import paddle
from paddle.autograd.functional import _check_tensors


def _product(t):
    if isinstance(t, int):
        return t
    else:
        return np.product(t)


def _get_item(t, idx):
    assert isinstance(t, paddle.Tensor), "The first argument t must be Tensor."
    assert isinstance(idx,
                      int), "The second argument idx must be an int number."
    flat_t = paddle.reshape(t, [-1])
    return flat_t.__getitem__(idx)


def _set_item(t, idx, value):
    assert isinstance(t, paddle.Tensor), "The first argument t must be Tensor."
    assert isinstance(idx,
                      int), "The second argument idx must be an int number."
    flat_t = paddle.reshape(t, [-1])
    flat_t.__setitem__(idx, value)
    return paddle.reshape(flat_t, t.shape)


def _compute_numerical_jacobian(func, xs, delta, np_dtype):
    xs = _check_tensors(xs, "xs")
    ys = _check_tensors(func(*xs), "ys")
    fin_size = len(xs)
    fout_size = len(ys)
    jacobian = list([] for _ in range(fout_size))
    for i in range(fout_size):
        jac_i = list([] for _ in range(fin_size))
        for j in range(fin_size):
            jac_i[j] = np.zeros(
                (_product(ys[i].shape), _product(xs[j].shape)), dtype=np_dtype)
        jacobian[i] = jac_i

    for j in range(fin_size):
        for q in range(_product(xs[j].shape)):
            orig = _get_item(xs[j], q)
            x_pos = orig + delta
            xs[j] = _set_item(xs[j], q, x_pos)
            ys_pos = _check_tensors(func(*xs), "ys_pos")

            x_neg = orig - delta
            xs[j] = _set_item(xs[j], q, x_neg)
            ys_neg = _check_tensors(func(*xs), "ys_neg")

            xs[j] = _set_item(xs[j], q, orig)

            for i in range(fout_size):
                for p in range(_product(ys[i].shape)):
                    y_pos = _get_item(ys_pos[i], p)
                    y_neg = _get_item(ys_neg[i], p)
                    jacobian[i][j][p][q] = (y_pos - y_neg) / delta / 2.
    return jacobian


def _compute_numerical_hessian(func, xs, delta, np_dtype):
    xs = _check_tensors(xs, "xs")
    ys = _check_tensors(func(*xs), "ys")
    fin_size = len(xs)
    hessian = list([] for _ in range(fin_size))
    for i in range(fin_size):
        hessian_i = list([] for _ in range(fin_size))
        for j in range(fin_size):
            hessian_i[j] = np.zeros(
                (_product(xs[i].shape), _product(xs[j].shape)), dtype=np_dtype)
        hessian[i] = hessian_i

    for i in range(fin_size):
        for p in range(_product(xs[i].shape)):
            for j in range(fin_size):
                for q in range(_product(xs[j].shape)):
                    orig = _get_item(xs[j], q)
                    x_pos = orig + delta
                    xs[j] = _set_item(xs[j], q, x_pos)
                    jacobian_pos = _compute_numerical_jacobian(func, xs, delta,
                                                               np_dtype)
                    x_neg = orig - delta
                    xs[j] = _set_item(xs[j], q, x_neg)
                    jacobian_neg = _compute_numerical_jacobian(func, xs, delta,
                                                               np_dtype)
                    xs[j] = _set_item(xs[j], q, orig)
                    hessian[i][j][p][q] = (
                        jacobian_pos[0][i][0][p] - jacobian_neg[0][i][0][p]
                    ) / delta / 2.
    return hessian
