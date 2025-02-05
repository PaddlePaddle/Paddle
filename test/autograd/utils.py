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

import enum
import sys
import typing

import numpy as np

import paddle
from paddle.incubate.autograd.utils import as_tensors


##########################################################
# Finite Difference Utils
##########################################################
def _product(t):
    return int(np.prod(t))


def _get_item(t, idx):
    assert isinstance(
        t, paddle.base.framework.Variable
    ), "The first argument t must be Tensor."
    assert isinstance(
        idx, int
    ), "The second argument idx must be an int number."
    flat_t = paddle.reshape(t, [-1])
    return flat_t.__getitem__(idx)


def _set_item(t, idx, value):
    assert isinstance(
        t, paddle.base.framework.Variable
    ), "The first argument t must be Tensor."
    assert isinstance(
        idx, int
    ), "The second argument idx must be an int number."
    flat_t = paddle.reshape(t, [-1])
    flat_t.__setitem__(idx, value)
    return paddle.reshape(flat_t, t.shape)


def _compute_numerical_jacobian(func, xs, delta, np_dtype):
    xs = list(as_tensors(xs))
    ys = list(as_tensors(func(*xs)))
    fin_size = len(xs)
    fout_size = len(ys)
    jacobian = [[] for _ in range(fout_size)]
    for i in range(fout_size):
        jac_i = [[] for _ in range(fin_size)]
        for j in range(fin_size):
            jac_i[j] = np.zeros(
                (_product(ys[i].shape), _product(xs[j].shape)), dtype=np_dtype
            )
        jacobian[i] = jac_i

    for j in range(fin_size):
        for q in range(_product(xs[j].shape)):
            orig = _get_item(xs[j], q)
            orig = paddle.assign(orig)
            x_pos = orig + delta
            xs[j] = _set_item(xs[j], q, x_pos)
            ys_pos = as_tensors(func(*xs))

            x_neg = orig - delta
            xs[j] = _set_item(xs[j], q, x_neg)
            ys_neg = as_tensors(func(*xs))

            xs[j] = _set_item(xs[j], q, orig)

            for i in range(fout_size):
                for p in range(_product(ys[i].shape)):
                    y_pos = _get_item(ys_pos[i], p)
                    y_neg = _get_item(ys_neg[i], p)
                    jacobian[i][j][p][q] = (y_pos - y_neg) / delta / 2.0
    return jacobian


def _compute_numerical_hessian(func, xs, delta, np_dtype):
    xs = list(as_tensors(xs))
    ys = list(as_tensors(func(*xs)))
    fin_size = len(xs)
    hessian = [[] for _ in range(fin_size)]
    for i in range(fin_size):
        hessian_i = [[] for _ in range(fin_size)]
        for j in range(fin_size):
            hessian_i[j] = np.zeros(
                (_product(xs[i].shape), _product(xs[j].shape)), dtype=np_dtype
            )
        hessian[i] = hessian_i

    for i in range(fin_size):
        for p in range(_product(xs[i].shape)):
            for j in range(fin_size):
                for q in range(_product(xs[j].shape)):
                    orig = _get_item(xs[j], q)
                    orig = paddle.assign(orig)
                    x_pos = orig + delta
                    xs[j] = _set_item(xs[j], q, x_pos)
                    jacobian_pos = _compute_numerical_jacobian(
                        func, xs, delta, np_dtype
                    )
                    x_neg = orig - delta
                    xs[j] = _set_item(xs[j], q, x_neg)
                    jacobian_neg = _compute_numerical_jacobian(
                        func, xs, delta, np_dtype
                    )
                    xs[j] = _set_item(xs[j], q, orig)
                    hessian[i][j][p][q] = (
                        (jacobian_pos[0][i][0][p] - jacobian_neg[0][i][0][p])
                        / delta
                        / 2.0
                    )
    return hessian


def concat_to_matrix(xs, is_batched=False):
    """Concats a tuple of tuple of Jacobian/Hessian matrix into one matrix"""
    rows = []
    for i in range(len(xs)):
        rows.append(np.concatenate(list(xs[i]), -1))
    return np.concatenate(rows, 1) if is_batched else np.concatenate(rows, 0)


def _compute_numerical_batch_jacobian(
    func, xs, delta, np_dtype, merge_batch=True
):
    no_batch_jacobian = _compute_numerical_jacobian(func, xs, delta, np_dtype)
    xs = list(as_tensors(xs))
    ys = list(as_tensors(func(*xs)))
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
            if merge_batch:
                batch_jac_i_j = np.reshape(batch_jac_i_j, (out_size, -1))
            batch_jac_i.append(batch_jac_i_j)
        bat_jac.append(batch_jac_i)

    return bat_jac


def _compute_numerical_batch_hessian(func, xs, delta, np_dtype):
    xs = list(as_tensors(xs))
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
            np.stack((hessian_res[i], hessian_res[mid + i]), axis=0)
        )
    return hessian_result


def _compute_numerical_vjp(func, xs, v, delta, np_dtype):
    xs = as_tensors(xs)
    jacobian = np.array(_compute_numerical_jacobian(func, xs, delta, np_dtype))
    if v is None:
        v = [paddle.ones_like(x) for x in xs]
    flat_v = np.array([v_el.numpy().reshape(-1) for v_el in v])
    vjp = [np.zeros((_product(x.shape)), dtype=np_dtype) for x in xs]
    for j in range(len(xs)):
        for q in range(_product(xs[j].shape)):
            vjp[j][q] = np.sum(
                jacobian[:, j, :, q].reshape(flat_v.shape) * flat_v
            )
    vjp = [vjp[j].reshape(xs[j].shape) for j in range(len(xs))]
    return vjp


def _compute_numerical_vhp(func, xs, v, delta, np_dtype):
    xs = list(as_tensors(xs))
    hessian = np.array(_compute_numerical_hessian(func, xs, delta, np_dtype))
    flat_v = np.array([v_el.numpy().reshape(-1) for v_el in v])
    vhp = [np.zeros((_product(x.shape)), dtype=np_dtype) for x in xs]
    for j in range(len(xs)):
        for q in range(_product(xs[j].shape)):
            vhp[j][q] = np.sum(
                hessian[:, j, :, q].reshape(flat_v.shape) * flat_v
            )
    vhp = [vhp[j].reshape(xs[j].shape) for j in range(len(xs))]
    return vhp


##########################################################
# TestCases of different function.
##########################################################
def reduce(x):
    return paddle.sum(x)


def reduce_dim(x):
    return paddle.sum(x, axis=0)


def matmul(x, y):
    return paddle.matmul(x, y)


def mul(x, y):
    return x * y


def pow(x, y):
    return paddle.pow(x, y)


def o2(x, y):
    return paddle.multiply(x, y), paddle.matmul(x, y.t())


def unuse(x, y):
    return paddle.sum(x)


def nested(x):
    def inner(y):
        return x * y

    return inner


def square(x):
    return x * x


##########################################################
# Parameterized Test Utils.
##########################################################

TEST_CASE_NAME = 'suffix'


def place(devices, key='place'):
    """A Decorator for a class which will make the class running on different
    devices .

    Args:
        devices (Sequence[Paddle.CUDAPlace|Paddle.CPUPlace]): Device list.
        key (str, optional): Defaults to 'place'.
    """

    def decorate(cls):
        module = sys.modules[cls.__module__].__dict__
        raw_classes = {
            k: v for k, v in module.items() if k.startswith(cls.__name__)
        }

        for raw_name, raw_cls in raw_classes.items():
            for d in devices:
                test_cls = dict(raw_cls.__dict__)
                test_cls.update({key: d})
                new_name = raw_name + '.' + d.__class__.__name__
                module[new_name] = type(new_name, (raw_cls,), test_cls)
            del module[raw_name]
        return cls

    return decorate


def parameterize(fields, values=None):
    """Decorator for a unittest class which make the class running on different
    test cases.

    Args:
        fields (Sequence): The field name sequence of test cases.
        values (Sequence, optional): The test cases sequence. Defaults to None.

    """
    fields = [fields] if isinstance(fields, str) else fields
    params = [dict(zip(fields, vals)) for vals in values]

    def decorate(cls):
        test_cls_module = sys.modules[cls.__module__].__dict__
        for i, values in enumerate(params):
            test_cls = dict(cls.__dict__)
            values = {
                k: staticmethod(v) if callable(v) else v
                for k, v in values.items()
            }
            test_cls.update(values)
            name = cls.__name__ + str(i)
            name = (
                name + '.' + values.get('suffix')
                if values.get('suffix')
                else name
            )

            test_cls_module[name] = type(name, (cls,), test_cls)

        for m in list(cls.__dict__):
            if m.startswith("test"):
                delattr(cls, m)
        return cls

    return decorate


##########################################################
# Utils for transpose different Jacobian/Hessian matrix format.
##########################################################

# B is batch size, N is row size, M is column size.
MatrixFormat = enum.Enum('MatrixFormat', ('NBM', 'BNM', 'NMB', 'NM'))


def _np_transpose_matrix_format(src, src_format, des_format):
    """Transpose Jacobian/Hessian matrix format."""
    supported_format = (MatrixFormat.NBM, MatrixFormat.BNM, MatrixFormat.NMB)
    if src_format not in supported_format or des_format not in supported_format:
        raise ValueError(
            f"Supported Jacobian format is {supported_format}, but got src: {src_format}, des: {des_format}"
        )

    src_axis = {c: i for i, c in enumerate(src_format.name)}
    dst_axis = tuple(src_axis[c] for c in des_format.name)

    return np.transpose(src, dst_axis)


def _np_concat_matrix_sequence(src, src_format=MatrixFormat.NM):
    """Convert a sequence of sequence of Jacobian/Hessian matrix into one huge
    matrix."""

    def concat_col(xs):
        if src_format in (MatrixFormat.NBM, MatrixFormat.BNM, MatrixFormat.NM):
            return np.concatenate(xs, axis=-1)
        else:
            return np.concatenate(xs, axis=1)

    def concat_row(xs):
        if src_format in (MatrixFormat.NBM, MatrixFormat.NM, MatrixFormat.NMB):
            return np.concatenate(xs, axis=0)
        else:
            return np.concatenate(xs, axis=1)

    supported_format = (
        MatrixFormat.NBM,
        MatrixFormat.BNM,
        MatrixFormat.NMB,
        MatrixFormat.NM,
    )
    if src_format not in supported_format:
        raise ValueError(
            f"Supported Jacobian format is {supported_format}, but got {src_format}"
        )
    if not isinstance(src, typing.Sequence):
        return src
    if not isinstance(src[0], typing.Sequence):
        src = [src]

    return concat_row(tuple(concat_col(xs) for xs in src))


##########################################################
# Utils for generating test data.
##########################################################
def gen_static_data_and_feed(xs, v, stop_gradient=True):
    feed = {}
    if isinstance(xs, typing.Sequence):
        static_xs = []
        for i, x in enumerate(xs):
            x = paddle.static.data(f"x{i}", x.shape, x.dtype)
            x.stop_gradient = stop_gradient
            static_xs.append(x)
        feed.update({f'x{idx}': value for idx, value in enumerate(xs)})
    else:
        static_xs = paddle.static.data('x', xs.shape, xs.dtype)
        static_xs.stop_gradient = stop_gradient
        feed.update({'x': xs})

    if isinstance(v, typing.Sequence):
        static_v = []
        for i, e in enumerate(v):
            e = paddle.static.data(f'v{i}', e.shape, e.dtype)
            e.stop_gradient = stop_gradient
            static_v.append(e)
        feed.update({f'v{i}': value for i, value in enumerate(v)})
    elif v is not None:
        static_v = paddle.static.data('v', v.shape, v.dtype)
        static_v.stop_gradient = stop_gradient
        feed.update({'v': v})
    else:
        static_v = v

    return feed, static_xs, static_v


def gen_static_inputs_and_feed(xs, stop_gradient=True):
    feed = {}
    if isinstance(xs, typing.Sequence):
        static_xs = []
        for i, x in enumerate(xs):
            x = paddle.static.data(f"x{i}", x.shape, x.dtype)
            x.stop_gradient = stop_gradient
            static_xs.append(x)
        feed.update({f'x{idx}': value for idx, value in enumerate(xs)})
    else:
        static_xs = paddle.static.data('x', xs.shape, xs.dtype)
        static_xs.stop_gradient = stop_gradient
        feed.update({'x': xs})
    return feed, static_xs
