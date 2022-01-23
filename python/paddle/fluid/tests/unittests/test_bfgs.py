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
import paddle
from paddle.incubate.optimizer.functional import bfgs_iterates, bfgs_minimize
from paddle.incubate.optimizer.functional.bfgs import (
    SearchState, verify_symmetric_positive_definite_matrix,
    update_approx_inverse_hessian)
from paddle.incubate.optimizer.functional.bfgs_utils import (vjp, vnorm_inf)

import tensorflow as tf


def jacfn_gen(f, create_graph=False):
    r"""Returns a helper function for computing the jacobians.
    
    Requires `f` to be single valued function. Batch input is allowed.

    The returned function, when called, returns a tensor of [Batched] gradients.
    """

    def jac(x):
        fval, grads = vjp(f, x, create_graph=create_graph)
        return grads

    return jac


def hesfn_gen(f):
    r"""Returns a helper function for computing the hessians.

    Requires `f` to be single valued function. Batch input is allowed.

    The returned function, when called, returns a tensor of [Batched]
    second order partial derivatives.
    """

    def hess(x):
        y = f(x)
        batch_mode = len(x.shape) > 1
        dim = x.shape[-1]
        vs = []
        for i in range(dim):
            v = paddle.zeros_like(x)
            if batch_mode:
                v[..., i] = 1
            else:
                v[i] = 1
            v = v.detach()
            vs.append(v)
        jacfn = jacfn_gen(f, create_graph=True)
        rows = [vjp(jacfn, x, v)[1] for v in vs]
        h = paddle.stack(rows, axis=-2)
        return h

    return hess


def verify_symmetric(H):
    perm = list(range(len(H.shape)))
    perm[-2:] = perm[-1], perm[-2]
    # batch_deltas = paddle.norm(H - H.transpose(perm), axis=perm[-2:])
    # is_symmetic = paddle.sum(batch_deltas) == 0.0
    assert paddle.allclose(H, H.transpose(perm)), (
        f"(Batched) matrix {H} is not symmetric.")


@paddle.no_grad()
def update_inv_hessian_strict(bat, H, s, y):
    dtype = H.dtype
    dim = s.shape[-1]
    if bat:
        rho = 1. / paddle.einsum('...i, ...i', s, y)
    else:
        rho = 1. / paddle.dot(s, y)
    rho = rho.unsqueeze(-1).unsqueeze(-1) if bat else rho.unsqueeze(-1)
    I = paddle.eye(dim, dtype=dtype)
    sy = paddle.einsum('...ij,...i,...j->...ij', rho, s, y)
    l = I - sy
    L = paddle.cholesky(H)
    lL = paddle.matmul(l, L)
    Lr = paddle.einsum('...ij->...ji', lL)
    lHr = paddle.matmul(lL, Lr)
    verify_symmetric(lHr)
    rsTs = paddle.einsum('...ij,...i,...j->...ij', rho, s, s)

    H_next = lHr + rsTs
    verify_symmetric_positive_definite_matrix(H_next)

    return H_next


def quadratic_gen(shape, dtype):
    center = paddle.rand(shape, dtype=dtype)
    hessian_shape = shape + shape[-1:]
    rotation = paddle.rand(hessian_shape, dtype=dtype)
    # hessian = paddle.einsum('...ik, ...jk', rotation, rotation)
    hessian = paddle.matmul(rotation, rotation, transpose_y=True)

    if shape[-1] > 1:
        verify_symmetric_positive_definite_matrix(hessian)
    else:
        hessian = paddle.abs(hessian)
        f = lambda x: paddle.sum((x - center) * hessian.squeeze(-1) * (x - center), axis=-1)
        return f, center

    def f(x):
        # (TODO:Tongxin) einsum may internally rely on dy2static which
        # does not support higher order gradients. 
        # y = paddle.einsum('...i, ...ij, ...j',
        #                   x - center,
        #                   hessian,
        #                   x - center)
        leftprod = paddle.matmul(hessian, (x - center).unsqueeze(-1))
        y = paddle.matmul((x - center).unsqueeze(-2), leftprod)
        if len(shape) > 1:
            y = y.reshape(shape[:-1])
        else:
            y = y.reshape([1])
        return y

    return f, center


class TestBFGS(unittest.TestCase):
    def setUp(self):
        pass

    def gen_configs(self):
        dtypes = ['float32', 'float64']
        shapes = {
            # '1d2v': [2],
            '2d2v': [2, 2],
            '1d50v': [50],
            '10d10v': [10, 10],
            '1d1v': [1],
            '2d1v': [2, 1],
        }
        for dtype in dtypes:
            for shape in shapes.values():
                yield shape, dtype

    def test_update_approx_inverse_hessian(self):
        paddle.seed(1234)
        for shape, dtype in self.gen_configs():
            bat = len(shape) > 1
            # only supports shapes with up to 2 dims.
            f, center = quadratic_gen(shape, dtype)
            # x0 = paddle.ones(shape, dtype=dtype)
            x0 = paddle.rand(shape, dtype=dtype)

            # The true inverse hessian value at x0
            hess = hesfn_gen(f)(x0)
            verify_symmetric_positive_definite_matrix(hess)
            hess_np = hess.numpy()
            hess_np_inv = np.linalg.inv(hess_np)
            h0 = paddle.to_tensor(hess_np_inv)

            verify_symmetric_positive_definite_matrix(h0)
            f0, g0 = vjp(f, x0)
            gnorm = vnorm_inf(f0)
            state = SearchState(bat, x0, f0, g0, h0, gnorm)

            # Verifies the two estimated invese Hessians are close
            for _ in range(5):
                s = paddle.rand(shape, dtype=dtype)
                x1 = x0 + s
                f1, g1 = vjp(f, x1)
                y = g1 - g0

                h1 = update_approx_inverse_hessian(state, h0, s, y)
                h1_strict = update_inv_hessian_strict(bat, h0, s, y)
                verify_symmetric_positive_definite_matrix(h1)
                verify_symmetric_positive_definite_matrix(h1_strict)

                self.assertTrue(True)

    def test_quadratic(self):
        paddle.seed(12345)
        for shape, dtype in self.gen_configs():
            f, center = quadratic_gen(shape, dtype)
            print(f'center {center}')
            print(f'f {f(center)}')
            x0 = paddle.ones(shape, dtype=dtype)
            result = bfgs_minimize(f, x0, dtype=dtype, iters=100, ls_iters=100)
            print(result)
            self.assertTrue(paddle.all(result.converged))
            self.assertTrue(paddle.allclose(result.x_location, center))


# shape = [2]
# dtype = 'float32'

# center = np.random.rand(2)
# scales = np.array([0.2, 1.5])
# x0 = np.ones_like(center)
# s = np.array([-0.5, -0.5])

# # scales = paddle.square(paddle.rand(shape, dtype=dtype))

# # # TF results as reference
# # center_tf = tf.convert_to_tensor(center)
# # x0_tf = tf.convert_to_tensor(x0)
# # s_tf = tf.convert_to_tensor(s)

# # def f_tf(x):
# #     return tf.reduce_sum(tf.square(x - center_tf), axis=-1)

# # with tf.GradientTape() as tape:
# #     tape.watch(x0_tf)
# #     y = f_tf(x0_tf)

# # g0_tf = tape.gradient(y, x0_tf)

# # x1_tf = x0_tf + s_tf
# # with tf.GradientTape() as tape:
# #     tape.watch(x1_tf)
# #     y = f_tf(x1_tf)

# # g1_tf = tape.gradient(y, x1_tf)

# # y_tf = g1_tf - g0_tf
# # h0_tf = tf.linalg.inv(tf.linalg.diag(g0_tf))
# # normalization_factor = tf.tensordot(s_tf, y_tf, 1)
# # h1_tf = tf_inv_hessian_update(y_tf, s_tf, normalization_factor, h0_tf)

# # Applies the proper update rules.
# center_pp = paddle.to_tensor(center)
# scales_pp = paddle.to_tensor(scales)
# x0_pp = paddle.to_tensor(x0)
# s_pp = paddle.to_tensor(s)

# def f(x):
#     return paddle.sum(paddle.square(x - center_pp), axis=-1)

# f0_pp, g0_pp = vjp(f, x0_pp)
# h0_pp = paddle.inverse(paddle.diag(g0_pp))
# state = SearchState(x0_pp, f0_pp, g0_pp, h0_pp)

# x1_pp = x0_pp + s_pp
# f1_pp, g1_pp = vjp(f, x1_pp)
# y_pp = g1_pp - g0_pp

# h1_pp = update_approx_inverse_hessian(state, h0_pp, s_pp, y_pp)

# h1_pp_proper = update_inv_hessian_strict(h0_pp, s_pp, y_pp)

if __name__ == "__main__":
    unittest.main()
