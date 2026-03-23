# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
CPU unit test for LarsMomentumKernel (forward).
(paddle/phi/kernels/cpu/lars_momentum_kernel.cc)

Kernel formulas (rescale_grad=1.0):
  p_norm   = norm2(p)
  g_norm   = norm2(g)
  local_lr = lr * lars_coeff * p_norm / (g_norm + wd * p_norm + eps)
             (only when wd > 0 AND p_norm > 0 AND g_norm > 0;
              otherwise local_lr = lr)
  v_out    = v * mu + local_lr * (g + wd * p)
  p_out    = p - v_out
"""

import unittest

import numpy as np

import paddle
from paddle.incubate.optimizer import LarsMomentumOptimizer


def _lars_ref(p, v, g, lr, mu, lars_coeff, wd, eps, rescale_grad=1.0):
    """NumPy reference implementation of one LARS step."""
    rg = rescale_grad * g
    p_norm = np.linalg.norm(p)
    g_norm = np.linalg.norm(rg)
    if wd > 0 and p_norm > 0 and g_norm > 0:
        local_lr = lr * lars_coeff * p_norm / (g_norm + wd * p_norm + eps)
    else:
        local_lr = lr
    v_out = v * mu + local_lr * (rg + wd * p)
    p_out = p - v_out
    return p_out.astype(np.float32), v_out.astype(np.float32)


def _one_step(p_np, g_np, lr, mu, lars_coeff, wd, eps, rescale_grad=1.0):
    """
    Run one LarsMomentumOptimizer step in dygraph mode on CPU.
    Initial velocity is zero (default).
    Returns updated param numpy array.
    """
    paddle.disable_static()
    paddle.set_device('cpu')

    param = paddle.create_parameter(
        shape=p_np.shape,
        dtype='float32',
        default_initializer=paddle.nn.initializer.Assign(p_np),
    )
    opt = LarsMomentumOptimizer(
        learning_rate=lr,
        momentum=mu,
        lars_coeff=lars_coeff,
        lars_weight_decay=wd,
        epsilon=eps,
        rescale_grad=rescale_grad,
        parameter_list=[param],
    )
    param.grad = paddle.to_tensor(g_np.copy())
    opt.step()
    return param.numpy()


class TestLarsMomentumKernelCPU(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        paddle.set_device('cpu')

    def test_basic_lars_step(self):
        """Standard LARS update; initial velocity = 0."""
        p_np = np.array([1.0, 2.0, 3.0, 4.0], dtype='float32')
        g_np = np.array([0.1, 0.2, 0.3, 0.4], dtype='float32')
        lr, mu, lars_coeff, wd, eps = 0.1, 0.9, 0.001, 0.0001, 1e-4

        p_ref, _ = _lars_ref(
            p_np, np.zeros_like(p_np), g_np, lr, mu, lars_coeff, wd, eps
        )
        p_out = _one_step(p_np, g_np, lr, mu, lars_coeff, wd, eps)

        np.testing.assert_allclose(
            p_out,
            p_ref,
            rtol=1e-5,
            err_msg=f'p_out mismatch: expected {p_ref}, got {p_out}',
        )

    def test_zero_weight_decay_uses_raw_lr(self):
        """
        When wd=0, local_lr falls back to bare lr (no LARS scaling).
        v_out = lr * g  (mu=0 for simple verification)
        p_out = p - lr * g
        """
        p_np = np.array([3.0, -1.0, 2.0], dtype='float32')
        g_np = np.array([0.5, 0.5, 0.5], dtype='float32')
        lr, mu, lars_coeff, wd, eps = 0.01, 0.0, 0.5, 0.0, 1e-4

        p_ref, _ = _lars_ref(
            p_np, np.zeros_like(p_np), g_np, lr, mu, lars_coeff, wd, eps
        )
        p_out = _one_step(p_np, g_np, lr, mu, lars_coeff, wd, eps)

        np.testing.assert_allclose(p_out, p_ref, rtol=1e-5)

    def test_multi_param_groups(self):
        """
        Two separate parameters in one optimizer call.
        The kernel iterates over op_num; both params must be updated correctly.
        """
        paddle.disable_static()
        paddle.set_device('cpu')

        p1_np = np.array([1.0, 2.0], dtype='float32')
        g1_np = np.array([0.1, 0.2], dtype='float32')
        p2_np = np.array([3.0, 4.0, 5.0], dtype='float32')
        g2_np = np.array([0.3, 0.4, 0.5], dtype='float32')

        lr, mu, lars_coeff, wd, eps = 0.1, 0.9, 0.001, 0.0001, 1e-4

        p1_ref, _ = _lars_ref(
            p1_np, np.zeros_like(p1_np), g1_np, lr, mu, lars_coeff, wd, eps
        )
        p2_ref, _ = _lars_ref(
            p2_np, np.zeros_like(p2_np), g2_np, lr, mu, lars_coeff, wd, eps
        )

        param1 = paddle.create_parameter(
            shape=p1_np.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(p1_np),
        )
        param2 = paddle.create_parameter(
            shape=p2_np.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(p2_np),
        )
        opt = LarsMomentumOptimizer(
            learning_rate=lr,
            momentum=mu,
            lars_coeff=lars_coeff,
            lars_weight_decay=wd,
            epsilon=eps,
            parameter_list=[param1, param2],
        )
        param1.grad = paddle.to_tensor(g1_np.copy())
        param2.grad = paddle.to_tensor(g2_np.copy())
        opt.step()

        np.testing.assert_allclose(
            param1.numpy(),
            p1_ref,
            rtol=1e-5,
            err_msg=f'param1: expected {p1_ref}, got {param1.numpy()}',
        )
        np.testing.assert_allclose(
            param2.numpy(),
            p2_ref,
            rtol=1e-5,
            err_msg=f'param2: expected {p2_ref}, got {param2.numpy()}',
        )


if __name__ == '__main__':
    unittest.main()
