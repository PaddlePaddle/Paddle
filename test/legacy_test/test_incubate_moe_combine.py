# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import random
import unittest

import numpy as np
from ernie_utils.moe_layer_uneven import GateCombine

import paddle
import paddle.nn.functional as F
from paddle.incubate.nn.functional import moe_combine

os.environ["FLAGS_flash_attn_version"] = "v1"
os.environ["FLAGS_cudnn_deterministic"] = "1"
os.environ["FLAGS_embedding_deterministic"] = "1"


def combining(x, combine_weights, scatter_index, hard_gate=False):
    """
    Args:
        x: Tensor[seq, dim]
        combine_weights: [seq, k]
        scatter_index:  ** [seq, k] **

    Returns:
        y: Tensor[s, dim]
    """
    x_gatherd = F.embedding(scatter_index, x)  # [s,k,dim]
    if hard_gate:
        return x_gatherd.squeeze(-2)
    # logger.info(f'combinning: {combine_weights}')
    y = (combine_weights.unsqueeze(-1) * x_gatherd).sum(1)
    # y = paddle.matmul(combine_weights.unsqueeze(1), x_gatherd).squeeze()  # [s,1,k] @ [s,k,dim] -> [s,1,dim]
    return y


def baseline_result(
    x_numpy, combine_weights_numpy, scatter_index_numpy, grad_numpy
):
    """baseline_result"""
    scatter_index = paddle.to_tensor(scatter_index_numpy)
    x = paddle.to_tensor(x_numpy).cast("float32")
    x.stop_gradient = False

    combine_weights = paddle.to_tensor(combine_weights_numpy).cast("float32")
    combine_weights.stop_gradient = False

    scatter_index = paddle.to_tensor(scatter_index_numpy)
    grad = paddle.to_tensor(grad_numpy).cast("float32")

    y = combining(x, combine_weights, scatter_index)
    paddle.autograd.backward([y], [grad], True)
    return [x.grad, combine_weights.grad, y]


def test_moe_combine(
    x_numpy, combine_weights_numpy, scatter_index_numpy, grad_numpy
):
    """baseline_result"""
    x = paddle.to_tensor(x_numpy).cast("float32")
    x.stop_gradient = False

    combine_weights = paddle.to_tensor(combine_weights_numpy).cast("float32")
    combine_weights.stop_gradient = False

    scatter_index = paddle.to_tensor(scatter_index_numpy).cast("int32")
    grad = paddle.to_tensor(grad_numpy).cast("float32")

    y = GateCombine.apply(x, combine_weights, scatter_index)
    paddle.autograd.backward([y], [grad], True)
    # grad.backward()
    return [x.grad, combine_weights.grad, y]


def gen_test_case(S, K, Dim, capacity_factor, seed=1234):
    """gen_test_case"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    x_numpy = np.random.rand(int(S * capacity_factor), Dim).astype(np.float32)
    combine_weights_numpy = np.random.rand(S, K).astype(np.float32)
    scatter_index_numpy = np.random.permutation(max(x_numpy.shape[0], S * K))[
        : S * K
    ].astype("int64")
    scatter_index_numpy = scatter_index_numpy.reshape([S, K])

    combine_weights_numpy[scatter_index_numpy >= x_numpy.shape[0]] = 0
    scatter_index_numpy[scatter_index_numpy >= x_numpy.shape[0]] = 0
    grad_numpy = np.random.randn(S, Dim).astype(np.float32)
    return x_numpy, combine_weights_numpy, scatter_index_numpy, grad_numpy


def testing(test_case):
    """testing"""
    [bl_x_grad, bl_combine_weights_grad, bl_y] = baseline_result(*test_case)
    [fused_x_grad, fused_combine_weights_grad, fused_y] = test_moe_combine(
        *test_case
    )
    np.testing.assert_allclose(
        fused_y.astype("float32").numpy(),
        bl_y.astype("float32").numpy(),
        err_msg="fwd precision not pass",
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        fused_x_grad.astype("float32").numpy(),
        bl_x_grad.astype("float32").numpy(),
        rtol=1e-6,
        err_msg="bwd grad precision not pass",
    )
    np.testing.assert_allclose(
        fused_combine_weights_grad.astype("float32").numpy(),
        bl_combine_weights_grad.astype("float32").numpy(),
        rtol=1e-6,
    )


class TestFused(unittest.TestCase):
    @unittest.skipIf(moe_combine is None, "test_moe_combine not installed")
    def test_cap_lt_2(
        self,
    ):
        """
        测试精度对齐的功能

        Args:
            无参，没有任何参数。

        Returns:
            NoneType：测试通过时返回None；测试失败时抛出异常。

        """
        testing(gen_test_case(S=1024, K=2, Dim=4096, capacity_factor=1.8))

    @unittest.skipIf(moe_combine is None, "test_moe_combine not installed")
    def test_cap_eq_2(
        self,
    ):
        """
        测试精度对齐的功能

        Args:
            无参，没有任何参数。

        Returns:
            NoneType：测试通过时返回None；测试失败时抛出异常。

        """
        testing(gen_test_case(S=1024, K=2, Dim=4096, capacity_factor=2))

    @unittest.skipIf(moe_combine is None, "test_moe_combine not installed")
    def test_cap_gt_2(
        self,
    ):
        """
        测试精度对齐的功能

        Args:
            无参，没有任何参数。

        Returns:
            NoneType：测试通过时返回None；测试失败时抛出异常。

        """
        testing(gen_test_case(S=1024, K=2, Dim=4096, capacity_factor=2.2))

    @unittest.skipIf(moe_combine is None, "test_moe_combine not installed")
    def test_k_gt_2(
        self,
    ):
        """
        测试精度对齐的功能

        Args:
            无参，没有任何参数。

        Returns:
            NoneType：测试通过时返回None；测试失败时抛出异常。

        """
        testing(gen_test_case(S=1024, K=8, Dim=4096, capacity_factor=2))


if __name__ == "__main__":

    unittest.main()
