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
from paddle.autograd.functional import _tensors


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
    xs = _tensors(xs, "xs")
    ys = _tensors(func(*xs), "ys")
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
            ys_pos = _tensors(func(*xs), "ys_pos")

            x_neg = orig - delta
            xs[j] = _set_item(xs[j], q, x_neg)
            ys_neg = _tensors(func(*xs), "ys_neg")

            xs[j] = _set_item(xs[j], q, orig)

            for i in range(fout_size):
                for p in range(_product(ys[i].shape)):
                    y_pos = _get_item(ys_pos[i], p)
                    y_neg = _get_item(ys_neg[i], p)
                    jacobian[i][j][p][q] = (y_pos - y_neg) / delta / 2.
    return jacobian


def _compute_numerical_hessian(func, xs, delta, np_dtype):
    xs = _tensors(xs, "xs")
    ys = _tensors(func(*xs), "ys")
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


def _compute_numerical_batch_jacobian(func, xs, delta, np_dtype):
    no_batch_jacobian = _compute_numerical_jacobian(func, xs, delta, np_dtype)
    xs = _tensors(xs, "xs")
    ys = _tensors(func(*xs), "ys")
    fin_size = len(xs)
    fout_size = len(ys)
    bs = xs[0].shape[0]
    bat_jac = []
    for i in range(fout_size):
        batch_jac_i = []
        for j in range(fin_size):
            jac = no_batch_jacobian[i][j]
            jac_shape = jac.shape
            out_size = jac_shape[0] // bs
            in_size = jac_shape[1] // bs
            jac = np.reshape(jac, (bs, out_size, bs, in_size))
            batch_jac_i_j = np.zeros(shape=(out_size, bs, in_size))
            for p in range(out_size):
                for b in range(bs):
                    for q in range(in_size):
                        batch_jac_i_j[p][b][q] = jac[b][p][b][q]
            batch_jac_i_j = np.reshape(batch_jac_i_j, (out_size, -1))
            batch_jac_i.append(batch_jac_i_j)
        bat_jac.append(batch_jac_i)

    return bat_jac


def _compute_numerical_batch_hessian(func, xs, delta, np_dtype):
    xs = _tensors(xs, "xs")
    batch_size = xs[0].shape[0]
    fin_size = len(xs)
    hessian = []
    for b in range(batch_size):
        x_l = []
        for j in range(fin_size):
            x_l.append(paddle.reshape(xs[j][b], shape=[1, -1]))
        hes_b = _compute_numerical_hessian(func, x_l, delta, np_dtype)
        if fin_size == 1:
            hessian.append(hes_b[0][0])
        else:
            hessian.append(hes_b)

    hessian_res = []
    for index in range(fin_size):
        x_reshape = paddle.reshape(xs[index], shape=[batch_size, -1])
        for index_ in range(fin_size):
            for i in range(x_reshape.shape[1]):
                tmp = []
                for j in range(batch_size):
                    if fin_size == 1:
                        tmp.extend(hessian[j][i])
                    else:
                        tmp.extend(hessian[j][i][index_][index])
                hessian_res.append(tmp)
        if fin_size == 1:
            return hessian_res

    hessian_result = []
    mid = len(hessian_res) // 2
    for i in range(mid):
        hessian_result.append(
            np.stack(
                (hessian_res[i], hessian_res[mid + i]), axis=0))
    return hessian_result


def _compute_numerical_vjp(func, xs, v, delta, np_dtype):
    xs = _tensors(xs, "xs")
    jacobian = np.array(_compute_numerical_jacobian(func, xs, delta, np_dtype))
    flat_v = np.array([v_el.numpy().reshape(-1) for v_el in v])
    vjp = [np.zeros((_product(x.shape)), dtype=np_dtype) for x in xs]
    for j in range(len(xs)):
        for q in range(_product(xs[j].shape)):
            vjp[j][q] = np.sum(jacobian[:, j, :, q].reshape(flat_v.shape) *
                               flat_v)
    vjp = [vjp[j].reshape(xs[j].shape) for j in range(len(xs))]
    return vjp


def _compute_numerical_vhp(func, xs, v, delta, np_dtype):
    xs = _tensors(xs, "xs")
    hessian = np.array(_compute_numerical_hessian(func, xs, delta, np_dtype))
    flat_v = np.array([v_el.numpy().reshape(-1) for v_el in v])
    vhp = [np.zeros((_product(x.shape)), dtype=np_dtype) for x in xs]
    for j in range(len(xs)):
        for q in range(_product(xs[j].shape)):
            vhp[j][q] = np.sum(hessian[:, j, :, q].reshape(flat_v.shape) *
                               flat_v)
    vhp = [vhp[j].reshape(xs[j].shape) for j in range(len(xs))]
    return vhp
